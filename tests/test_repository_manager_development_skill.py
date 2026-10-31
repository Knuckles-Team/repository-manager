"""PIPE-CONNSPEC-R157.4: the repository-manager development skill is packaged."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = (
    ROOT
    / "repository_manager"
    / "skills"
    / "repository-manager-development"
    / "SKILL.md"
)
PYPROJECT = ROOT / "pyproject.toml"

pytestmark = pytest.mark.spec("PIPE-CONNSPEC-R157.4")


def test_development_skill_has_rapid_contract_and_package_binding() -> None:
    """The central criterion names a real, discoverable, source-backed skill."""

    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\nname: repository-manager-development\n")
    assert "## Rapid delivery: the 5-minute contract" in text
    assert "repository_manager/repository_manager.py" in text
    assert "repository_manager/mcp_server.py" in text
    assert "repository-manager-development-lifecycle" in text
    assert "repository-manager-lane-lifecycle" in text
    assert "rm_lane_start" in text
    assert "rm_lane_doctor" in text
    assert "rm_lane_finish" in text
    assert "rm_merge_queue_status" in text

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]["repository_manager"]
    assert "skills/**" in package_data
    assert "scion_sdk.skill_providers" not in project["project"]["entry-points"]
