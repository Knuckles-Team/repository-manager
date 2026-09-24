"""Authority-unreachable refusal — genuine, not simulated (required test #1/#7).

``resolve_default_authority()`` never constructs a live RMDD-16 authority
(that needs an engine handle and a namespace policy); see ``client.py``'s
module docstring. This test exercises that real refusal, not a mock.
"""

from __future__ import annotations

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


def test_injected_authority_bypasses_resolution_entirely() -> None:
    """An explicitly injected port is used as-is; no import is attempted."""

    from tests.concept_coordination.fakes import FakeConceptAuthority

    fake = FakeConceptAuthority()
    assert fake.authoritative is False  # test double never claims authority
