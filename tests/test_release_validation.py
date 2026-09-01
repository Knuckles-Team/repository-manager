"""Closed-schema probes for release metadata and manifest repository URLs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from repository_manager.release_validation import (
    canonical_repository_url,
    read_release_document,
    repository_name,
    validate_release_document,
)


def _complete_document() -> dict:
    return {
        "project": {
            "name": "agent-one",
            "version": "1.2.3",
            "description": "One agent",
            "readme": {"text": "README", "content-type": "text/markdown"},
            "requires-python": ">=3.11",
            "license": "MIT",
            "license-files": ["LICENSE*"],
            "authors": [{"name": "Author", "email": "author@example.invalid"}],
            "maintainers": [{"name": "Maintainer"}],
            "keywords": ["agents"],
            "classifiers": ["Programming Language :: Python :: 3"],
            "urls": {"Homepage": "https://example.invalid"},
            "scripts": {"agent-one": "agent_one:main"},
            "gui-scripts": {"agent-one-gui": "agent_one:gui"},
            "entry-points": {"agent.plugins": {"one": "agent_one:plugin"}},
            "dependencies": ["packaging>=24"],
            "optional-dependencies": {"test": ["pytest>=8"]},
            "dynamic": [],
        },
        "build-system": {
            "requires": ["hatchling>=1"],
            "build-backend": "hatchling.build",
            "backend-path": ["backend"],
        },
    }


def test_complete_standardized_pep621_schema_is_accepted() -> None:
    assert validate_release_document(_complete_document(), expected_name="agent-one")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", b"agent-one"),
        ("version", 1),
        ("description", []),
        ("readme", 1),
        ("requires-python", []),
        ("license", 1),
        ("license-files", "LICENSE"),
        ("authors", {}),
        ("maintainers", ["Maintainer"]),
        ("keywords", "agents"),
        ("classifiers", {}),
        ("urls", []),
        ("scripts", []),
        ("gui-scripts", []),
        ("entry-points", []),
        ("dependencies", "packaging"),
        ("optional-dependencies", []),
        ("dynamic", "version"),
    ],
)
def test_every_standardized_project_key_is_strictly_typed(
    field: str, value: object
) -> None:
    document = _complete_document()
    document["project"][field] = value

    assert not validate_release_document(document, expected_name="agent-one")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readme", {"file": "README.md", "bogus": "x"}),
        ("license", {"file": "LICENSE", "text": "MIT"}),
        ("authors", [{"name": "A", "role": "owner"}]),
        ("authors", [{"email": "not-an-email"}]),
        ("maintainers", [{"email": 1}]),
        ("scripts", {"agent-one": 1}),
        ("scripts", {"agent-one": "not a reference"}),
        ("gui-scripts", {1: "agent_one:gui"}),
        ("entry-points", {"agent.plugins": {"one": 1}}),
        ("urls", {"Homepage": 1}),
        ("optional-dependencies", {"test": [1]}),
        ("dependencies", ["not a requirement @@@"]),
        ("license", "not a valid SPDX expression"),
    ],
)
def test_nested_pep621_shapes_fail_closed(field: str, value: object) -> None:
    document = _complete_document()
    document["project"][field] = value

    assert not validate_release_document(document, expected_name="agent-one")


def test_unknown_and_unsupported_dynamic_project_keys_fail_closed() -> None:
    unknown = _complete_document()
    unknown["project"]["private"] = True
    assert not validate_release_document(unknown, expected_name="agent-one")

    dynamic_license_files = _complete_document()
    del dynamic_license_files["project"]["license-files"]
    dynamic_license_files["project"]["dynamic"] = ["license-files"]
    assert not validate_release_document(
        dynamic_license_files, expected_name="agent-one"
    )

    conflict = deepcopy(_complete_document())
    conflict["project"]["dynamic"] = ["description"]
    assert not validate_release_document(conflict, expected_name="agent-one")


@pytest.mark.parametrize(
    "url",
    [
        b"https://example.invalid/org/agent-one.git",
        " https://example.invalid/org/agent-one.git",
        "https://example.invalid/org/agent-one.git\x7f",
        "https://example.invalid/org/agent-one.git\x85",
        "https://user@example.invalid/org/agent-one.git",
        "https://user:secret@example.invalid/org/agent-one.git",
        "https://example.invalid:/org/agent-one.git",
        "https://example.invalid:nope/org/agent-one.git",
        "https://example.invalid:0/org/agent-one.git",
        "https://example.invalid:65536/org/agent-one.git",
        "ssh://example.invalid/org/agent-one.git",
        "git@example.invalid:org/agent-one.git",
        "https:///org/agent-one.git",
        "https://example.invalid/org/agent-one",
        "https://example.invalid/org/agent-one.git?token=secret",
        "https://example.invalid/org/agent-one.git?",
        "https://example.invalid/org/agent-one.git#fragment",
        "https://example.invalid/org/agent-one.git#",
    ],
)
def test_repository_url_rejects_noncanonical_authority_and_raw_text(
    url: object,
) -> None:
    with pytest.raises(ValueError):
        canonical_repository_url(url)


def test_repository_url_canonicalization_and_name_share_one_parser() -> None:
    url = "HTTPS://Example.Invalid:443/Knuckles%2DTeam/%61gent-one.git"

    assert canonical_repository_url(url) == (
        "https://example.invalid/Knuckles-Team/agent-one.git"
    )
    assert repository_name(url) == "agent-one"


def test_release_file_reader_rejects_invalid_encoding_toml_and_schema(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pyproject.toml"
    for content in (
        b"\xff\xfe[project]",
        b"[project\nname='agent-one'",
        b"[project]\nname='agent-one'\nunknown=true\n",
    ):
        manifest.write_bytes(content)
        assert read_release_document(manifest, expected_name="agent-one") is None


def _file_metadata_document(readme_path: str, license_files: str = "LICENSE") -> str:
    return f"""\
[project]
name = "agent-one"
version = "1.0"
license = "MIT"
license-files = ["{license_files}"]

[project.readme]
file = "{readme_path}"
content-type = "text/markdown"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""


@pytest.mark.parametrize("readme_path", ["../README.md", "/tmp/README.md"])
def test_release_file_reader_rejects_escaping_readme_paths(
    tmp_path: Path, readme_path: str
) -> None:
    (tmp_path / "LICENSE").write_text("MIT")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(_file_metadata_document(readme_path))

    assert read_release_document(manifest, expected_name="agent-one") is None


def test_release_file_reader_rejects_symlinked_readme_and_license_paths(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "README.md").write_text("outside")
    (outside / "LICENSE").write_text("MIT")
    (tmp_path / "README.md").symlink_to(outside / "README.md")
    (tmp_path / "LICENSE").symlink_to(outside / "LICENSE")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(_file_metadata_document("README.md"))

    assert read_release_document(manifest, expected_name="agent-one") is None


def test_release_file_reader_rejects_symlinked_license_table_file(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-license-outside"
    outside.mkdir()
    (outside / "LICENSE").write_text("MIT")
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "LICENSE").symlink_to(outside / "LICENSE")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """\
[project]
name = "agent-one"
version = "1.0"
license = {file = "LICENSE"}
readme = {file = "README.md", content-type = "text/markdown"}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    )

    assert read_release_document(manifest, expected_name="agent-one") is None


def test_release_file_reader_requires_readme_content_type_and_safe_license_glob(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "LICENSE").write_text("MIT")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(_file_metadata_document("README.md", "../LICENSE"))
    assert read_release_document(manifest, expected_name="agent-one") is None

    document = _complete_document()
    document["project"]["readme"] = {"file": "README.md"}
    assert not validate_release_document(document, expected_name="agent-one")
