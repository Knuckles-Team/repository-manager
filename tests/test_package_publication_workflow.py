"""Keep source validation independent from tag-authorized publication."""

from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.spec("RM-RELEASE-R008")


def workflow(name):
    """Read workflow event names without YAML 1.1's boolean 'on' coercion."""
    return yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("refs/tags/v3.5.2", True),
        ("refs/tags/v4.0.0", True),
        ("refs/tags/3.5.2", False),
        ("refs/heads/main", False),
        ("refs/heads/v3.5.2", False),
    ],
)
def test_publication_only_matches_version_tag_pushes(ref, expected):
    events = workflow("pipeline.yml")["on"]
    assert set(events) == {"push"}
    push = events["push"]
    assert set(push) == {"tags"}
    matches = ref.startswith("refs/tags/") and any(
        fnmatchcase(ref.removeprefix("refs/tags/"), pattern)
        for pattern in push["tags"]
    )
    assert matches is expected


def test_source_ci_still_validates_main_and_pull_requests():
    events = workflow("ci.yml")["on"]
    assert "pull_request" in events
    assert events["push"]["branches"] == ["main"]


def test_publication_delegates_exact_source_guard_and_requires_tag():
    jobs = workflow("pipeline.yml")["jobs"]
    python = jobs["publish-pypi"]
    assert python["uses"] == (
        "Knuckles-Team/pipelines/.github/workflows/python_pipeline.yml@main"
    )
    # No custom source ref, steps, or alternate publisher can bypass the shared
    # workflow's frozen-source and conflicting-tag checks.
    assert set(python) == {"uses", "with", "permissions", "secrets"}
    assert python["with"] == {"require-tag": "true"}
    assert python["secrets"] == {"PYPI_API_TOKEN": "${{ secrets.PYPI_API_TOKEN }}"}
    assert jobs["publish-docker"]["needs"] == "publish-pypi"
    assert jobs["publish-docker"]["secrets"] == {
        name: "${{ secrets." + name + " }}"
        for name in (
            "DOCKER_REGISTRY",
            "DOCKER_USERNAME",
            "DOCKER_PASSWORD",
            "DOCKER_REPOSITORY",
        )
    }
