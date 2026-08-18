"""Focused tests for generic uv sibling materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

import repository_manager.repository_manager as repository_manager_module
from repository_manager.repository_manager import Git


def _manager(tmp_path: Path, *names: str) -> tuple[Git, dict[str, Path]]:
    projects = {name: tmp_path / name for name in ("consumer", *names)}
    for path in projects.values():
        path.mkdir()
    manager = Git(path=str(tmp_path))
    manager.project_map = {
        f"local://{name}": str(path) for name, path in projects.items()
    }
    return manager, projects


def _manifest(project: Path, *source_names: str) -> None:
    entries = "\n".join(
        f'{name} = {{ path = ".uv-workspace-siblings/{name}", editable = true }}'
        for name in source_names
    )
    (project / "pyproject.toml").write_text(
        f"[tool.uv.sources]\n{entries}\n", encoding="utf-8"
    )


def test_materializes_every_declared_sibling_without_an_allowlist(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper-alpha", "helper-beta")
    _manifest(projects["consumer"], "helper-alpha", "helper-beta")

    names = manager._materialize_uv_siblings(str(projects["consumer"]))

    assert names == ("helper-alpha", "helper-beta")
    for name in names:
        link = projects["consumer"] / ".uv-workspace-siblings" / name
        assert link.is_symlink()
        assert link.resolve() == projects[name].resolve()


def test_accepts_pep503_equivalent_source_and_workspace_names(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper-pkg")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        "helper_pkg = { path = '.uv-workspace-siblings/helper-pkg' }\n",
        encoding="utf-8",
    )

    assert manager._materialize_uv_siblings(str(projects["consumer"])) == (
        "helper-pkg",
    )
    assert (
        projects["consumer"] / ".uv-workspace-siblings" / "helper-pkg"
    ).resolve() == projects["helper-pkg"].resolve()


def test_rejects_source_alias_before_creating_sibling_registration(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources]\nalias = { path = '.uv-workspace-siblings/helper' }\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert not (projects["consumer"] / ".uv-workspace-siblings").exists()


def test_rejects_unknown_source_field_without_partial_mutation(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        "helper = { path = '.uv-workspace-siblings/helper', mystery = true }\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown field"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert not (projects["consumer"] / ".uv-workspace-siblings").exists()


def test_rejects_local_remote_conflict_before_partial_mutation(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        "helper = { path = '.uv-workspace-siblings/helper', git = 'https://example.invalid/helper.git' }\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting source kinds"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert not (projects["consumer"] / ".uv-workspace-siblings").exists()


def test_validates_every_source_alternative_before_writing(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        "helper = [\n"
        "  { path = '.uv-workspace-siblings/helper' },\n"
        "  { path = '.uv-workspace-siblings/not-helper' },\n"
        "]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert not (projects["consumer"] / ".uv-workspace-siblings").exists()


def test_dot_source_is_reserved_for_the_owning_project(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[project]\nname = 'consumer'\n\n[tool.uv.sources]\nhelper = { path = '.' }\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="owning project"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert not (projects["consumer"] / ".uv-workspace-siblings").exists()


def test_dot_source_for_owning_project_is_validated_without_a_link(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[project]\nname = 'consumer'\n\n"
        "[tool.uv.sources]\n"
        "consumer = { path = '.' }\n"
        "helper = { path = '.uv-workspace-siblings/helper' }\n",
        encoding="utf-8",
    )

    assert manager._materialize_uv_siblings(str(projects["consumer"])) == ("helper",)


def test_rejects_canonical_directory_with_another_package_identity(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["helper"] / "pyproject.toml").write_text(
        "[project]\nname = 'different-package'\n", encoding="utf-8"
    )
    _manifest(projects["consumer"], "helper")

    with pytest.raises(ValueError, match="project name.*match"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_remote_source_alternative_is_checked_but_not_materialized(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        "helper = [\n"
        "  { path = '.uv-workspace-siblings/helper' },\n"
        "  { git = 'https://example.invalid/helper.git', branch = 'main' },\n"
        "]\n",
        encoding="utf-8",
    )

    assert manager._materialize_uv_siblings(str(projects["consumer"])) == ("helper",)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "helper pkg",
        "helper/pkg",
        "éhelper",
        "_helper",
        "helper_",
        "-helper",
        "helper-",
        "helper..",
    ],
)
def test_rejects_names_outside_ascii_pep503_grammar(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="ASCII PEP 503"):
        Git._normalize_uv_name(invalid_name, label="test name")


@pytest.mark.parametrize("invalid_kind", ["source", "project", "path"])
def test_rejects_invalid_source_project_and_path_identities(
    tmp_path: Path, invalid_kind: str
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    if invalid_kind == "source":
        manifest = (
            "[tool.uv.sources]\n"
            '"helper name" = { path = ".uv-workspace-siblings/helper" }\n'
        )
    elif invalid_kind == "project":
        manifest = (
            "[project]\nname = 'consumer name'\n\n"
            "[tool.uv.sources]\n"
            "consumer = { path = '.' }\n"
        )
    else:
        manifest = (
            "[tool.uv.sources]\n"
            'helper = { path = ".uv-workspace-siblings/helper name" }\n'
        )
    (projects["consumer"] / "pyproject.toml").write_text(manifest, encoding="utf-8")

    with pytest.raises(ValueError, match="ASCII PEP 503"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_failed_second_publish_restores_all_prior_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, projects = _manager(tmp_path, "helper-alpha", "helper-beta")
    _manifest(projects["consumer"], "helper-alpha", "helper-beta")
    sibling_dir = projects["consumer"] / ".uv-workspace-siblings"
    sibling_dir.mkdir()
    old_alpha = tmp_path / "old-alpha"
    old_beta = tmp_path / "old-beta"
    old_alpha.mkdir()
    old_beta.mkdir()
    links = {
        "helper-alpha": sibling_dir / "helper-alpha",
        "helper-beta": sibling_dir / "helper-beta",
    }
    links["helper-alpha"].symlink_to(old_alpha)
    links["helper-beta"].symlink_to(old_beta)
    before = {name: links[name].readlink() for name in links}

    original_replace = repository_manager_module.os.replace
    replace_calls = 0

    def fail_second_publish(source: str, destination: str) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected second publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(repository_manager_module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected second publish"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert replace_calls >= 2
    assert {name: links[name].readlink() for name in links} == before
    assert sorted(path.name for path in sibling_dir.iterdir()) == sorted(links)
    assert not list(sibling_dir.glob(".*.tmp"))


def test_failed_publish_removes_new_empty_sibling_scaffolding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, projects = _manager(tmp_path, "helper-alpha", "helper-beta")
    _manifest(projects["consumer"], "helper-alpha", "helper-beta")
    original_replace = repository_manager_module.os.replace
    replace_calls = 0

    def fail_second_publish(source: str, destination: str) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected second publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(repository_manager_module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected second publish"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    sibling_dir = projects["consumer"] / ".uv-workspace-siblings"
    assert not sibling_dir.exists()


@pytest.mark.parametrize(
    "declared_path",
    [
        ".uv-workspace-siblings/../helper",
        "../helper",
        "/tmp/helper",
        ".uv-workspace-siblings\\helper",
        ".uv-workspace-siblings/helper/child",
        ".uv-workspace-siblings/.",
        ".uv-workspace-siblings/..",
    ],
)
def test_refuses_noncanonical_declared_sibling_path(
    tmp_path: Path, declared_path: str
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        f"[tool.uv.sources]\nhelper = {{ path = '{declared_path}' }}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="direct .* path"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_refuses_non_symlink_registration(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    _manifest(projects["consumer"], "helper")
    sibling_dir = projects["consumer"] / ".uv-workspace-siblings"
    sibling_dir.mkdir()
    (sibling_dir / "helper").mkdir()

    with pytest.raises(ValueError, match="non-symlink"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_rejects_non_symlink_registration_without_replacing_prior_links(
    tmp_path: Path,
) -> None:
    manager, projects = _manager(tmp_path, "helper-alpha", "helper-beta")
    _manifest(projects["consumer"], "helper-alpha", "helper-beta")
    sibling_dir = projects["consumer"] / ".uv-workspace-siblings"
    sibling_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    first = sibling_dir / "helper-alpha"
    first.symlink_to(outside)
    second = sibling_dir / "helper-beta"
    second.mkdir()

    with pytest.raises(ValueError, match="non-symlink"):
        manager._materialize_uv_siblings(str(projects["consumer"]))

    assert first.resolve() == outside.resolve()
    assert second.is_dir() and not second.is_symlink()


def test_corrects_wrong_symlink_target(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper", "wrong")
    _manifest(projects["consumer"], "helper")
    sibling_dir = projects["consumer"] / ".uv-workspace-siblings"
    sibling_dir.mkdir()
    link = sibling_dir / "helper"
    link.symlink_to(projects["wrong"], target_is_directory=True)

    manager._materialize_uv_siblings(str(projects["consumer"]))

    assert link.resolve() == projects["helper"].resolve()


def test_corrects_wrong_symlink_target_outside_workspace(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    _manifest(projects["consumer"], "helper")
    outside = tmp_path.parent / f"{tmp_path.name}-wrong-target"
    outside.mkdir()
    sibling_dir = projects["consumer"] / ".uv-workspace-siblings"
    sibling_dir.mkdir()
    link = sibling_dir / "helper"
    link.symlink_to(outside, target_is_directory=True)

    manager._materialize_uv_siblings(str(projects["consumer"]))

    assert link.resolve() == projects["helper"].resolve()


def test_repeated_materialization_is_idempotent(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    _manifest(projects["consumer"], "helper")

    manager._materialize_uv_siblings(str(projects["consumer"]))
    link = projects["consumer"] / ".uv-workspace-siblings" / "helper"
    first_link_text = link.readlink()
    first_inode = link.lstat().st_ino

    manager._materialize_uv_siblings(str(projects["consumer"]))

    assert link.readlink() == first_link_text
    assert link.lstat().st_ino == first_inode


def test_missing_canonical_target_fails_closed(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path)
    _manifest(projects["consumer"], "missing")

    with pytest.raises(ValueError, match="missing from the workspace map"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_refuses_project_path_outside_approved_root(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    _manifest(projects["consumer"], "helper")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-consumer"
    outside.mkdir()
    (outside / "pyproject.toml").write_text(
        '[tool.uv.sources]\nhelper = { path = ".uv-workspace-siblings/helper" }\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="project path escapes workspace root"):
        manager._materialize_uv_siblings(str(outside))


def test_refuses_symlinked_project_path(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    _manifest(projects["consumer"], "helper")
    alias = tmp_path / "consumer-alias"
    alias.symlink_to(projects["consumer"], target_is_directory=True)

    with pytest.raises(ValueError, match="project path contains symlink component"):
        manager._materialize_uv_siblings(str(alias))


def test_refuses_symlinked_canonical_target(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path)
    _manifest(projects["consumer"], "helper")
    outside = tmp_path.parent / f"{tmp_path.name}-helper-real"
    outside.mkdir()
    linked_target = tmp_path / "helper"
    linked_target.symlink_to(outside, target_is_directory=True)
    manager.project_map["local://helper"] = str(linked_target)

    with pytest.raises(ValueError, match="canonical sibling target.*symlink component"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_refuses_symlinked_canonical_target_ancestor(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path)
    _manifest(projects["consumer"], "helper")
    real_parent = tmp_path / "canonical-real"
    real_parent.mkdir()
    (real_parent / "helper").mkdir()
    linked_parent = tmp_path / "canonical-link"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    manager.project_map["local://helper"] = str(linked_parent / "helper")

    with pytest.raises(ValueError, match="canonical sibling target.*symlink component"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_refuses_symlinked_sibling_directory(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    _manifest(projects["consumer"], "helper")
    outside = tmp_path.parent / f"{tmp_path.name}-sibling-dir"
    outside.mkdir()
    (projects["consumer"] / ".uv-workspace-siblings").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(ValueError, match="symlink sibling directory"):
        manager._materialize_uv_siblings(str(projects["consumer"]))


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("tool = 'invalid'\n", r"\[tool\] must be a table"),
        ("[tool]\nuv = 'invalid'\n", r"\[tool\.uv\] must be a table"),
        (
            "[tool.uv]\nsources = 'invalid'\n",
            r"\[tool\.uv\.sources\] must be a table",
        ),
        (
            "[tool.uv.sources]\nhelper = 'invalid'\n",
            "must be a table or list of tables",
        ),
        (
            "[tool.uv.sources]\nhelper = ['invalid']\n",
            "must contain only tables",
        ),
        (
            "[tool.uv.sources]\nhelper = []\n",
            "must contain at least one table",
        ),
        (
            "[tool.uv.sources]\nhelper = { path = 42 }\n",
            "has a non-string path",
        ),
    ],
)
def test_rejects_malformed_uv_source_config(
    tmp_path: Path, manifest: str, message: str
) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(manifest, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        manager._materialize_uv_siblings(str(projects["consumer"]))


def test_rejects_malformed_uv_source_manifest(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        "[tool.uv.sources\nhelper = {}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot parse uv source manifest"):
        manager._materialize_uv_siblings(str(projects["consumer"]))
