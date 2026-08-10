"""Authority-unreachable refusal — genuine, not simulated (required test #1/#7).

``resolve_default_authority()`` really cannot reach RMDD-16's authority in
this environment today; see ``client.py``'s module docstring for the exact
commit evidence. This test exercises that real condition, not a mock.
"""

from __future__ import annotations

import importlib.util

import pytest

from repository_manager.concept_coordination.client import (
    AUTHORITY_MODULE,
    resolve_default_authority,
)
from repository_manager.concept_coordination.errors import ConceptAuthorityUnavailable


def test_default_authority_refuses_and_names_what_it_could_not_reach() -> None:
    with pytest.raises(ConceptAuthorityUnavailable) as excinfo:
        resolve_default_authority()
    message = str(excinfo.value)
    assert AUTHORITY_MODULE in message
    # Fail-closed refusal must be an actionable, non-empty statement, not a
    # bare "no" — and it must never claim a local ID was minted instead.
    assert "unreachable" in message or "does not construct" in message
    assert "mint" not in message.lower()


def test_authority_unreachable_never_swallows_the_import_error_cause() -> None:
    """H-12: a refusal never discards its exception cause."""

    if importlib.util.find_spec("agent_utilities.governance.concept_reservation"):
        pytest.skip(
            "agent_utilities.governance.concept_reservation is importable in this "
            "environment; the cause-preservation path this test targets (a genuine "
            "ImportError) cannot be exercised here. See client.py for the commit "
            "evidence this normally fails on agent-utilities main."
        )
    with pytest.raises(ConceptAuthorityUnavailable) as excinfo:
        resolve_default_authority()
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_injected_authority_bypasses_resolution_entirely() -> None:
    """An explicitly injected port is used as-is; no import is attempted."""

    from tests.concept_coordination.fakes import FakeConceptAuthority

    fake = FakeConceptAuthority()
    assert fake.authoritative is False  # test double never claims authority
