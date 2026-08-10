"""The narrow authority boundary RMDD-17's action core depends on.

Mirrors the public method surface of RMDD-16's
``agent_utilities.governance.concept_reservation.ConceptReservationAuthority``
Protocol (``reserve``/``get``/``list``/``transition``) structurally — as a
:class:`typing.Protocol` with no import of agent-utilities — so
repository-manager never takes a hard, load-bearing dependency on
agent-utilities internals. Any object exposing these four methods (the real
native authority once injected by RMDD-20's entrypoint, an integration-test
double, or a future graph-os MCP adapter) satisfies this port.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ConceptClaimRecord, ConceptClaimRequest
from .state import ConceptClaimState, ConceptClaimVisibility

__all__ = ["ConceptAuthorityPort"]


@runtime_checkable
class ConceptAuthorityPort(Protocol):
    """Structural port a real or fake concept authority must satisfy.

    ``authoritative`` mirrors RMDD-16's own convention: a fixture/test double
    advertises ``authoritative = False`` and must never be treated as a
    production cross-host authority (see ``client.py``).
    """

    authoritative: bool

    def reserve(self, request: ConceptClaimRequest) -> ConceptClaimRecord:
        """Create-if-absent a claim for ``request.concept_id``.

        A repeated ``(tenant_ref, request_key_ref, concept_id)`` replays the
        original record when the immutable input is unchanged, and raises a
        conflict when it differs — request-key idempotency, never a second
        allocation.
        """
        ...

    def get(self, reservation_id: str, *, tenant_ref: str) -> ConceptClaimRecord: ...

    def list(
        self,
        *,
        tenant_ref: str,
        namespace: str | None = None,
        state: ConceptClaimState | None = None,
        concept_prefix: str | None = None,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> tuple[list[ConceptClaimRecord], str | None]: ...

    def transition(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
        target: ConceptClaimState,
        visibility: ConceptClaimVisibility | None = None,
    ) -> ConceptClaimRecord:
        """Fenced lifecycle mutation; a stale ``expected_fence`` refuses."""
        ...
