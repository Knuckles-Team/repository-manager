"""CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R006: the pack extends only the core ontology."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.spec("CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R006")

ONTOLOGY = Path(__file__).resolve().parents[1] / "repository_manager" / "ontology"
CORE_NAMES = ("Worktree", "archived", "lastActivityAt")


def test_pack_declares_no_core_term() -> None:
    text = "\n".join(p.read_text() for p in ONTOLOGY.glob("*.ttl"))
    for name in CORE_NAMES:
        assert not re.search(rf"^:{name}\s", text, re.M), (
            f"pack redeclares core term {name}"
        )
        assert not re.search(rf"(?<![\w:]):{name}(?![\w-])", text), (
            f"pack references :{name}"
        )
    assert "core:Worktree" in text
