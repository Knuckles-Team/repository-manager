"""Concept-claim lifecycle states, mirrored from RMDD-16's authority contract.

RMDD-16 (``agent_utilities.governance.concept_reservation``,
``ConceptReservationState``/``ConceptReservationVisibility``) is the owning
authority for this state machine.  RMDD-17 does not redefine or widen it —
this module mirrors the exact wire values so repository-manager can reason
about a claim's lifecycle without an import-time dependency on a module
that, as of this lane (2026-08-10), exists only on ``agent-utilities``
feature branches (``rmdd-program-integration-0808``,
``rmdd-28-native-lane-authority-0809``, ``rmdd-19-provenance-0810``) and
backup ref ``refs/lane-backup/rmdd-16-concept-authority-0809`` (commit
``bbb09765``) — not yet on ``agent-utilities`` ``main`` (see
``client.py`` for the full account and the exact commits checked).

``tests/concept_coordination/test_state.py`` asserts these values against the
real enum whenever ``agent_utilities.governance.concept_reservation`` *is*
importable, so drift is caught the moment both repositories' ``main``
branches carry the module — the test degrades to a documented skip only when
the module is genuinely absent, never a silent pass.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ConceptClaimState",
    "ConceptClaimVisibility",
    "TERMINAL_STATES",
    "VISIBILITY_RANK",
    "is_legal_claim_transition",
    "minimum_visibility_for_target",
    "strongest_visibility",
]


class ConceptClaimState(StrEnum):
    """Durable claim lifecycle states (mirrors RMDD-16 exactly)."""

    RESERVED = "reserved"
    MATERIALIZED = "materialized"
    LANDED = "landed"
    RELEASED = "released"
    EXPIRED = "expired"
    TOMBSTONED = "tombstoned"


class ConceptClaimVisibility(StrEnum):
    """The strongest visibility reached by a claim (mirrors RMDD-16 exactly)."""

    PRIVATE = "private"
    FRAGMENT = "fragment"
    REPOSITORY = "repository"
    EXTERNAL = "external"


TERMINAL_STATES: frozenset[ConceptClaimState] = frozenset(
    {ConceptClaimState.TOMBSTONED}
)

_TRANSITIONS: dict[ConceptClaimState, frozenset[ConceptClaimState]] = {
    ConceptClaimState.RESERVED: frozenset(
        {
            ConceptClaimState.MATERIALIZED,
            ConceptClaimState.RELEASED,
            ConceptClaimState.EXPIRED,
            ConceptClaimState.TOMBSTONED,
        }
    ),
    ConceptClaimState.MATERIALIZED: frozenset(
        {
            ConceptClaimState.LANDED,
            ConceptClaimState.RELEASED,
            ConceptClaimState.EXPIRED,
            ConceptClaimState.TOMBSTONED,
        }
    ),
    ConceptClaimState.LANDED: frozenset({ConceptClaimState.TOMBSTONED}),
    ConceptClaimState.RELEASED: frozenset({ConceptClaimState.TOMBSTONED}),
    ConceptClaimState.EXPIRED: frozenset({ConceptClaimState.TOMBSTONED}),
    ConceptClaimState.TOMBSTONED: frozenset(),
}


def is_legal_claim_transition(
    current: ConceptClaimState, target: ConceptClaimState
) -> bool:
    """Mirror RMDD-16's authority transition map; never widen it locally."""

    return target in _TRANSITIONS.get(current, frozenset())


VISIBILITY_RANK: dict[ConceptClaimVisibility, int] = {
    ConceptClaimVisibility.PRIVATE: 0,
    ConceptClaimVisibility.FRAGMENT: 1,
    ConceptClaimVisibility.REPOSITORY: 2,
    ConceptClaimVisibility.EXTERNAL: 3,
}

# Every lifecycle transition never lowers visibility, and reaching a given
# state floors it at a minimum (mirrors RMDD-16's authority exactly, read
# from ``agent_utilities.governance.concept_reservation`` commit ``bbb09765``
# — see ``client.py`` for why that module cannot be imported here):
# materializing is at least fragment-visible, landing is at least
# repository-visible, and a tombstone is always fully external.
_MINIMUM_VISIBILITY_FOR_TARGET: dict[ConceptClaimState, ConceptClaimVisibility] = {
    ConceptClaimState.MATERIALIZED: ConceptClaimVisibility.FRAGMENT,
    ConceptClaimState.LANDED: ConceptClaimVisibility.REPOSITORY,
    ConceptClaimState.TOMBSTONED: ConceptClaimVisibility.EXTERNAL,
}


def strongest_visibility(*values: ConceptClaimVisibility) -> ConceptClaimVisibility:
    """Return the highest-ranked visibility among ``values``; never regresses."""

    if not values:
        raise ValueError("strongest_visibility requires at least one value")
    return max(values, key=VISIBILITY_RANK.__getitem__)


def minimum_visibility_for_target(
    target: ConceptClaimState,
) -> ConceptClaimVisibility | None:
    """Return the visibility floor a transition to ``target`` enforces, if any."""

    return _MINIMUM_VISIBILITY_FOR_TARGET.get(target)
