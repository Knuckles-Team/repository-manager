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




def test_source_ci_still_validates_main_and_pull_requests():
    events = workflow("ci.yml")["on"]
    assert "pull_request" in events
    assert events["push"]["branches"] == ["main"]
