"""Adversarial scope proofs for the Phase-5 PyPI agent release wave."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml  # type: ignore[import-untyped]

from repository_manager.repository_manager import Git, GitResult


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
        "epistemic": "https://example.invalid/epistemic-graph.git",
        "utilities": "https://example.invalid/agent-utilities.git",
        "webui": "https://example.invalid/agent-webui.git",
        "agent": "https://example.invalid/agent-one.git",
        "service": "https://example.invalid/service-one.git",
        "image": "https://example.invalid/image-one.git",
        "plans": "https://example.invalid/plans.git",
    }
    paths = {
        "pipeline": tmp_path / "pipelines",
        "epistemic": tmp_path / "epistemic-graph",
        "utilities": tmp_path / "agent-utilities",
        "webui": tmp_path / "agent-webui",
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
        urls["epistemic"]: (),
        urls["utilities"]: (),
        urls["webui"]: (),
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


def _target_names(targets: list[tuple[str, str]] | None) -> list[str] | None:
    return None if targets is None else [name for name, _path in targets]


def test_bulk_bump_plan_contains_only_manifest_agent_pypi_targets(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)

    phases, total = manager._build_bump_phase_list(
        config=_five_phase_config(), start_phase=5, filter_set=None
    )

    assert total == 1
    assert _target_names(phases[0]["targets"]) == ["agent-one"]


def test_bulk_precommit_scope_cannot_absorb_infrastructure(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)

    names = _target_names(manager._pre_commit_project_targets(_five_phase_config()))

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

    assert manager._auto_start_phase(_five_phase_config(), operation="bump") is None


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
    assert _target_names(bump_phases[0]["targets"]) == ["agent-one"]
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

    names = _target_names(
        manager._pre_commit_project_targets(
            _five_phase_config(), {"agent-one", "service-one"}
        )
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
        "[project]\nname='agent-one'\nversion='1.0'\ndynamic=['name']\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        "[project]\nname='agent-one'\nversion='1.0'\ndynamic=['bogus']\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        "[project]\nname='agent-one'\nversion='1.0'\ndescription='static'\n"
        "dynamic=['description']\n"
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


def test_supported_non_static_dynamic_field_remains_eligible(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    agent = tmp_path / "agent-packages" / "agents" / "agent-one"
    _write_release_metadata(
        agent,
        document=(
            "[project]\nname='agent-one'\nversion='1.0'\n"
            "dynamic=['description']\n"
            "[build-system]\nrequires=['hatchling']\n"
            "build-backend='hatchling.build'\n"
        ),
    )

    assert [name for name, _path in manager._bulk_release_targets(set())] == [
        "agent-one"
    ]


def test_invalid_toml_encoding_fails_closed(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    manifest = tmp_path / "agent-packages" / "agents" / "agent-one" / "pyproject.toml"
    manifest.write_bytes(b"\xff\xfe[project]")

    assert manager._bulk_release_targets(set()) == []


def test_bulk_target_rejects_workspace_escape(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/agent-one.git"
    outside = tmp_path.parent / "agent-one"
    _write_release_metadata(outside)
    manager.project_map[url] = str(outside)

    assert manager._bulk_release_targets(set()) == []


def test_bulk_target_rejects_lexical_parent_before_normalization(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/agent-one.git"
    aliases = tmp_path / "agent-packages" / "agents" / "aliases"
    aliases.mkdir(parents=True)
    manager.project_map[url] = str(aliases / ".." / "agent-one")
    manager._project_categories[url] = ("agent-packages", "agents")

    assert manager._bulk_release_targets(set()) == []
    with pytest.raises(ValueError, match="lexical parent"):
        manager._pre_commit_target_dirs([("agent-one", manager.project_map[url])])


def test_symlink_parent_traversal_cannot_widen_bulk_scope(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    category_root = tmp_path / "agent-packages" / "agents"
    external = tmp_path / "external" / "nested"
    _write_release_metadata(external / "arr-mcp")
    _write_release_metadata(category_root / "arr-mcp")
    link = category_root / "link"
    link.symlink_to(external, target_is_directory=True)
    url = "https://example.invalid/arr-mcp.git"
    manager.project_map[url] = str(link / ".." / "arr-mcp")
    manager._project_categories[url] = ("agent-packages", "agents")

    assert [name for name, _path in manager._bulk_release_targets(set())] == [
        "agent-one"
    ]
    with pytest.raises(ValueError, match="lexical parent"):
        manager._pre_commit_target_dirs([("arr-mcp", manager.project_map[url])])


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


def test_mutation_entrypoints_refuse_external_symlink_targets(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _scoped_manager(tmp_path)
    external = tmp_path / "external"
    _write_release_metadata(external)
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)
    git_action = MagicMock()
    monkeypatch.setattr(manager, "git_action", git_action)

    for operation in (
        lambda: manager.bump_version("patch", path=str(linked)),
        lambda: manager.push_project(path=str(linked)),
        lambda: manager.pre_commit(path=str(linked)),
        lambda: manager.commit_project("test", path=str(linked)),
        lambda: manager.add_project(path=str(linked)),
        lambda: manager.commit_code_project("test", path=str(linked)),
    ):
        result = operation()
        assert result.status == "error"
        assert result.error is not None
        assert "symlink component" in result.error.message

    git_action.assert_not_called()


def test_malformed_url_basename_fails_closed(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/...git"
    path = tmp_path / "agent-packages" / "agents" / ".."
    manager.project_map = {url: str(path)}
    manager._project_categories = {url: ("agent-packages", "agents")}

    assert manager._bulk_release_targets(set()) == []


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/org/../agent-one.git",
        "https://example.invalid/org/./agent-one.git",
        "https://example.invalid/org/%2e%2e/agent-one.git",
        "https://example.invalid/org/%2e/agent-one.git",
        "https://example.invalid/org//agent-one.git",
        "https://example.invalid/org/%2Fescape/agent-one.git",
        "https://example.invalid/org/%5cescape/agent-one.git",
        "https://example.invalid/org/%252Fescape/agent-one.git",
        "https://example.invalid/org/%ff/agent-one.git",
        "https://example.invalid/org/agent-one.git/",
    ],
)
def test_unsafe_url_path_segment_fails_closed(tmp_path: Path, url: str) -> None:
    manager = _scoped_manager(tmp_path)
    agent_path = tmp_path / "agent-packages" / "agents" / "agent-one"
    manager.project_map = {url: str(agent_path)}
    manager._project_categories = {url: ("agent-packages", "agents")}

    assert manager._bulk_release_targets(set()) == []


def test_safe_percent_encoded_url_segments_are_decoded(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/Knuckles%2DTeam/%61gent-one.git"
    agent_path = tmp_path / "agent-packages" / "agents" / "agent-one"
    manager.project_map = {url: str(agent_path)}
    manager._project_categories = {url: ("agent-packages", "agents")}

    assert [name for name, _path in manager._bulk_release_targets(set())] == [
        "agent-one"
    ]


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

    repeated_within = {"phases": [{"phase": 1, "projects": ["agent-one", "agent-one"]}]}
    with pytest.raises(ValueError, match="duplicate maintenance project name"):
        manager._build_bump_phase_list(
            config=repeated_within, start_phase=1, filter_set=None
        )

    repeated_across = {
        "phases": [
            {"phase": 1, "projects": ["agent-one"]},
            {"phase": 2, "project": "agent-one"},
        ]
    }
    with pytest.raises(ValueError, match="duplicate maintenance project name"):
        manager._build_push_phase_list(
            config=repeated_across, start_phase=1, project_filter=None
        )


def test_phase_exclude_is_identical_across_all_release_planners(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _scoped_manager(tmp_path)
    config = {
        "phases": [
            {
                "phase": 5,
                "bulk_bump": True,
                "bulk_push": True,
                "exclude": ["agent-*"],
            }
        ]
    }
    monkeypatch.setattr(manager, "_repo_has_pending_work", lambda _path: True)

    bump, bump_total = manager._build_bump_phase_list(
        config=config, start_phase=5, filter_set=None
    )
    push, push_total = manager._build_push_phase_list(
        config=config, start_phase=5, project_filter=None
    )

    assert (bump, bump_total) == ([], 0)
    assert (push, push_total) == ([], 0)
    assert manager._pre_commit_project_targets(config, start_phase=5) == []
    assert manager._auto_start_phase(config, operation="bump") is None


def test_excluded_bulk_target_can_enter_a_later_phase_consistently(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _scoped_manager(tmp_path)
    config = {
        "phases": [
            {
                "phase": 5,
                "bulk_bump": True,
                "bulk_push": True,
                "exclude": ["agent-one"],
            },
            {"phase": 6, "bulk_bump": True, "bulk_push": True},
        ]
    }
    monkeypatch.setattr(manager, "_repo_has_pending_work", lambda _path: True)

    bump, _ = manager._build_bump_phase_list(
        config=config, start_phase=5, filter_set=None
    )
    push, _ = manager._build_push_phase_list(
        config=config, start_phase=5, project_filter=None
    )

    assert [
        (phase["phase_num"], _target_names(phase["targets"])) for phase in bump
    ] == [(6, ["agent-one"])]
    assert [
        (phase["phase_num"], [name for name, _path in phase["projects_to_push"]])
        for phase in push
    ] == [(6, ["agent-one"])]
    assert _target_names(
        manager._pre_commit_project_targets(config, start_phase=5)
    ) == ["agent-one"]
    assert manager._auto_start_phase(config, operation="bump") == 6


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
    precommit_names = _target_names(
        manager._pre_commit_project_targets(config, start_phase=2, single_phase=True)
    )

    assert [phase["phase_num"] for phase in bump_phases] == [2]
    assert [phase["phase_num"] for phase in push_phases] == [2]
    assert precommit_names == ["service-one"]


def test_mixed_explicit_and_bulk_targets_have_identical_sorted_union(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)
    config = {
        "phases": [
            {
                "phase": 5,
                "projects": ["pipelines", "agent-one"],
                "bulk_bump": True,
                "bulk_push": True,
            }
        ]
    }

    bump, _ = manager._build_bump_phase_list(
        config=config, start_phase=5, filter_set=None
    )
    push, _ = manager._build_push_phase_list(
        config=config, start_phase=5, project_filter=None
    )
    precommit = _target_names(
        manager._pre_commit_project_targets(config, start_phase=5)
    )

    assert _target_names(bump[0]["targets"]) == ["agent-one", "pipelines"]
    assert [name for name, _path in push[0]["projects_to_push"]] == [
        "agent-one",
        "pipelines",
    ]
    assert precommit == ["agent-one", "pipelines"]


def test_target_order_does_not_depend_on_manifest_insertion_order(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)
    second_url = "https://example.invalid/agent-two.git"
    second_path = tmp_path / "agent-packages" / "agents" / "agent-two"
    _write_release_metadata(second_path)
    manager.project_map[second_url] = str(second_path)
    manager._project_categories[second_url] = ("agent-packages", "agents")
    config = {"phases": [{"phase": 5, "bulk_bump": True, "bulk_push": True}]}

    expected = ["agent-one", "agent-two"]
    first, _ = manager._build_bump_phase_list(
        config=config, start_phase=5, filter_set=None
    )
    manager.project_map = dict(reversed(list(manager.project_map.items())))
    second, _ = manager._build_bump_phase_list(
        config=config, start_phase=5, filter_set=None
    )

    assert (
        _target_names(first[0]["targets"])
        == _target_names(second[0]["targets"])
        == expected
    )


def test_auto_start_uses_first_nonempty_effective_reentry_phase(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _scoped_manager(tmp_path)
    second_url = "https://example.invalid/agent-two.git"
    second_path = tmp_path / "agent-packages" / "agents" / "agent-two"
    _write_release_metadata(second_path)
    manager.project_map[second_url] = str(second_path)
    manager._project_categories[second_url] = ("agent-packages", "agents")
    config = {
        "phases": [
            {"phase": 5, "bulk_bump": True, "exclude": ["agent-one"]},
            {"phase": 6, "bulk_bump": True},
        ]
    }
    monkeypatch.setattr(
        manager,
        "_repo_has_pending_work",
        lambda path: path.endswith("agent-one"),
    )

    assert manager._auto_start_phase(config, operation="bump") == 6


@pytest.mark.parametrize("reverse", [False, True])
def test_manifest_duplicate_repository_urls_fail_closed_order_independently(
    tmp_path: Path, reverse: bool
) -> None:
    url = "https://example.invalid/org/%61gent-one.git"
    canonical_duplicate = "https://example.invalid/org/agent-one.git"
    root_repositories = [{"url": canonical_duplicate}]
    nested_repositories = [{"url": url}]
    if reverse:
        root_repositories, nested_repositories = (
            nested_repositories,
            root_repositories,
        )
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "duplicate",
                "path": str(tmp_path / "workspace"),
                "repositories": root_repositories,
                "subdirectories": {
                    "agent-packages": {
                        "subdirectories": {
                            "agents": {"repositories": nested_repositories}
                        }
                    }
                },
            }
        )
    )
    manager = Git(path=str(tmp_path / "workspace"))

    assert manager.load_projects_from_yaml(str(manifest)) is False
    assert manager.project_map == {}
    assert manager._project_categories == {}


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"unknown": True},
        {"phases": None},
        {"phases": [{"name": "one", "phase": True}]},
        {"phases": [{"name": "one", "phase": 1, "bulk_bump": "true"}]},
        {"phases": [{"name": "one", "phase": 1, "wait_minutes": False}]},
        {"phases": [{"name": "one", "phase": 1, "wait_minutes": -0.1}]},
        {"phases": [{"name": "one", "phase": 1, "wait_minutes": float("nan")}]},
        {"phases": [{"name": "one", "phase": 1, "wait_minutes": float("inf")}]},
        {"phases": [{"name": "one", "phase": 1, "wait_minutes": 1440.1}]},
        {"phases": [{"name": "one", "phase": 1, "project": None}]},
        {
            "phases": [
                {
                    "name": "one",
                    "phase": 1,
                    "updates": [{"package": "pkg", "unknown": "value"}],
                }
            ]
        },
    ],
)
def test_raw_maintenance_config_rejects_coercion_unknowns_and_nulls(
    tmp_path: Path, config: object
) -> None:
    manager = _scoped_manager(tmp_path)

    assert manager._resolve_maintenance_config(config) is None


def test_barrier_metadata_uses_same_fail_closed_release_validator(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)
    agent_path = tmp_path / "agent-packages" / "agents" / "agent-one"
    (agent_path / "pyproject.toml").write_bytes(b"\xff\xfe[project]")

    assert manager._phase_published_packages([("agent-one", str(agent_path))]) == {}

    _write_release_metadata(
        agent_path,
        document=(
            "[project]\nname='agent-one'\nversion='1.0'\nunknown=true\n"
            "[build-system]\nrequires=['hatchling']\n"
            "build-backend='hatchling.build'\n"
        ),
    )
    assert manager._phase_published_packages([("agent-one", str(agent_path))]) == {}


def _collision_manager(tmp_path: Path) -> tuple[Git, list[tuple[str, str]]]:
    names = [
        "arr-mcp",
        "container-manager-mcp",
        "documentdb-mcp",
        "jellyfin-mcp",
        "kafka-mcp",
        "mealie-mcp",
        "searxng-mcp",
        "vector-mcp",
    ]
    manager = Git(path=str(tmp_path))
    expected: list[tuple[str, str]] = []
    for name in names:
        agent_path = tmp_path / "agent-packages" / "agents" / name
        service_path = tmp_path / "services" / name
        _write_release_metadata(agent_path)
        _write_release_metadata(service_path)
        agent_url = f"https://zzz.invalid/agents/{name}.git"
        service_url = f"https://aaa.invalid/services/{name}.git"
        manager.project_map[service_url] = str(service_path)
        manager.project_map[agent_url] = str(agent_path)
        manager._project_categories[service_url] = ("services",)
        manager._project_categories[agent_url] = ("agent-packages", "agents")
        expected.append((name, str(agent_path)))
    return manager, expected


def test_all_eight_agent_service_collisions_preserve_exact_paths_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    manager, expected = _collision_manager(tmp_path)
    config = {
        "phases": [
            {
                "name": "agents",
                "phase": 5,
                "bulk_bump": True,
                "bulk_push": True,
            }
        ]
    }

    bump, bump_total = manager._build_bump_phase_list(
        config=config, start_phase=5, filter_set=None
    )
    push, push_total = manager._build_push_phase_list(
        config=config, start_phase=5, project_filter=None
    )
    precommit = manager._pre_commit_project_targets(config, start_phase=5)

    assert bump_total == push_total == 8
    assert bump[0]["targets"] == push[0]["projects_to_push"] == precommit == expected
    assert not any("/services/" in path for _name, path in expected)

    pending_paths: list[str] = []

    def record_pending(path: str) -> bool:
        pending_paths.append(path)
        return True

    monkeypatch.setattr(manager, "_repo_has_pending_work", record_pending)
    assert manager._auto_start_phase(config, operation="bump") == 5
    assert pending_paths == [expected[0][1]]

    pre_commit_projects = MagicMock(return_value=[])
    commit_projects = MagicMock(return_value=[])
    monkeypatch.setattr(manager, "pre_commit_projects", pre_commit_projects)
    monkeypatch.setattr(manager, "commit_projects", commit_projects)
    manager._run_bump_pre_commit_stage(config, None, start_phase=5, single_phase=False)
    pre_commit_projects.assert_called_once_with(
        run=True,
        autoupdate=True,
        projects=[path for _name, path in expected],
    )

    bump_version = MagicMock(
        return_value=GitResult(status="success", data="new_version=1.0.1")
    )
    monkeypatch.setattr(manager, "bump_version", bump_version)
    monkeypatch.setattr(manager, "update_dependency", MagicMock(return_value=False))
    manager.phased_bumpversion(
        config=config, start_phase=5, auto_start=False, force=True
    )
    assert [call.kwargs["path"] for call in bump_version.call_args_list] == [
        path for _name, path in expected
    ]

    push_project = MagicMock(return_value=GitResult(status="success", data="Pushed"))
    monkeypatch.setattr(manager, "push_project", push_project)
    manager.phased_push(config=config, start_phase=5, auto_start=False)
    assert sorted(
        call.kwargs["path"] for call in push_project.call_args_list
    ) == sorted(path for _name, path in expected)


def test_auto_start_is_operation_specific_and_single_phase_never_advances(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _scoped_manager(tmp_path)
    config = {
        "phases": [
            {"name": "bump only", "phase": 5, "bulk_bump": True},
            {"name": "push only", "phase": 6, "bulk_push": True},
        ]
    }
    monkeypatch.setattr(manager, "_repo_has_pending_work", lambda _path: True)

    assert manager._auto_start_phase(config, operation="bump") == 5
    assert manager._auto_start_phase(config, operation="push") == 6

    push_project = MagicMock(return_value=GitResult(status="success", data="Pushed"))
    monkeypatch.setattr(manager, "push_project", push_project)
    manager.phased_push(config=config, start_phase=5, single_phase=True)
    push_project.assert_not_called()
