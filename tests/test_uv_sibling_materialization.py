"""Focused tests for generic uv sibling materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_refuses_traversal_in_declared_sibling_path(tmp_path: Path) -> None:
    manager, projects = _manager(tmp_path, "helper")
    (projects["consumer"] / "pyproject.toml").write_text(
        '[tool.uv.sources]\nhelper = { path = ".uv-workspace-siblings/../helper" }\n',
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
