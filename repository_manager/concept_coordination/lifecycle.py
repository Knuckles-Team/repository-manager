"""Landing/abort transition requests (deliverable 6).

Marks the claims backing a landed generation permanent, and releases only
the claims an aborted candidate is actually eligible to release. Never
releases or reuses a landed/externally-visible claim as part of an abort —
"abort cannot release landed/visible claim" is enforced here, not merely
documented.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .errors import ConceptClaimConflict
from .fragments import write_fragment_row
from .models import ConceptClaimRecord
from .port import ConceptAuthorityPort
from .state import ConceptClaimState

__all__ = ["abort_claims", "land_claims"]


def land_claims(
    authority: ConceptAuthorityPort,
    *,
    tenant_ref: str,
    owner_ref: str,
    records: Iterable[ConceptClaimRecord],
    repo_root: Path,
    lane_ref: str,
) -> list[ConceptClaimRecord]:
    """Transition every given claim to LANDED (permanent), fenced and idempotent.

    A claim already LANDED is returned unchanged rather than re-transitioned
    (matches RMDD-16's documented same-owner idempotent-replay rule).
    """

    landed: list[ConceptClaimRecord] = []
    for record in records:
        if record.state is ConceptClaimState.LANDED:
            landed.append(record)
            continue
        updated = authority.transition(
            record.reservation_id,
            tenant_ref=tenant_ref,
            owner_ref=owner_ref,
            expected_fence=record.fence,
            target=ConceptClaimState.LANDED,
        )
        write_fragment_row(repo_root, lane_ref, updated.canonical_payload())
        landed.append(updated)
    return landed


def abort_claims(
    authority: ConceptAuthorityPort,
    *,
    tenant_ref: str,
    owner_ref: str,
    records: Iterable[ConceptClaimRecord],
    repo_root: Path,
    lane_ref: str,
) -> list[ConceptClaimRecord]:
    """Release only never-visible claims; refuse to release a landed one.

    RESERVED/MATERIALIZED claims transition to RELEASED. A LANDED or
    TOMBSTONED claim raises :class:`ConceptClaimConflict` instead of being
    silently skipped or force-released — the caller must know its abort was
    only partially honored, never assume full rollback.
    """

    released: list[ConceptClaimRecord] = []
    for record in records:
        if record.state in (ConceptClaimState.LANDED, ConceptClaimState.TOMBSTONED):
            raise ConceptClaimConflict(
                f"cannot release a {record.state.value} claim: "
                f"{record.concept_id} is already landed/externally visible"
            )
        if record.state is ConceptClaimState.RELEASED:
            released.append(record)
            continue
        updated = authority.transition(
            record.reservation_id,
            tenant_ref=tenant_ref,
            owner_ref=owner_ref,
            expected_fence=record.fence,
            target=ConceptClaimState.RELEASED,
        )
        write_fragment_row(repo_root, lane_ref, updated.canonical_payload())
        released.append(updated)
    return released
