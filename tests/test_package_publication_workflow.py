"""Keep source validation independent from tag-authorized publication."""

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


def test_source_ci_still_validates_main_and_pull_requests():
    connector = workflow("connector.yml")
    events = connector["on"]
    assert events["pull_request"]["branches"] == ["main"]
    assert events["push"]["branches"] == ["main"]
    # Publication is routed by the shared pipeline only for pushed v-prefixed tags.
    assert events["push"]["tags"] == ["v*"]
    assert connector["jobs"]["pipeline"]["uses"] == (
        "Knuckles-Team/pipelines/.github/workflows/connector_pipeline.yml@main"
    )


def test_connector_workflow_is_the_only_source_ci_caller():
    names = {p.name for p in (ROOT / ".github" / "workflows").glob("*.yml")}
    assert names - {"windows-ci.yml"} == {"connector.yml"}
