"""Injected concept-authority client resolution (RMDD-17 work item 1).

RMDD-17 never allocates concept IDs itself — it is a thin, MCP/CLI-neutral
consumer of RMDD-16's central authority. This module's only job is to hand
the action core a :class:`~repository_manager.concept_coordination.port.ConceptAuthorityPort`:

* **The normal path is injection.** RMDD-20's MCP/CLI entrypoint (or a test)
  constructs the real authority — wired to a live epistemic-graph engine
  handle and this repository's namespace policy, which is RMDD-16/RMDD-23
  scope, not RMDD-17's — and passes it into
  :class:`~repository_manager.concept_coordination.action_core.ConceptCoordinationActions`
  directly. RMDD-17 owns no engine wiring and constructs no live authority.
* **When nothing is injected**, :func:`resolve_default_authority` is the
  fallback the action core calls. It never fabricates a local allocator; it
  attempts the same lazy ``try/except ImportError`` pattern this codebase
  uses for every optional dependency (repository-manager ``AGENTS.md``,
  "optional dependency" guardrail), and always raises
  :class:`~repository_manager.concept_coordination.errors.ConceptAuthorityUnavailable`
  naming exactly what could not be reached — because even a successful
  import cannot yield a *live* authority without an engine handle this
  module deliberately does not construct.

**Why this always refuses today, verified 2026-08-10.** The authority module
(``agent_utilities/governance/concept_reservation.py``) is real and complete
(``ConceptReservationService``, ``NativeConceptReservationAuthority``,
``FixtureConceptReservationAuthority``, states
reserved/materialized/landed/released/expired/tombstoned, fenced
transitions, ``reconcile_projection`` — merged at agent-utilities commit
``bbb09765``, "feat(governance): add native concept reservation authority")
— but that commit is **not** an ancestor of agent-utilities ``main``
(verified: ``git merge-base --is-ancestor bbb09765 <agent-utilities main>``
returns false at agent-utilities ``main`` = ``338226f5``). It exists only on
branches ``rmdd-program-integration-0808``, ``rmdd-28-native-lane-authority-0809``,
``rmdd-19-provenance-0810``, and backup ref
``refs/lane-backup/rmdd-16-concept-authority-0809``. The RMDD-16 ledger row
records it "CLOSED" with tests passing "post-merge" — that merge landed on
the AU integration branch, never on AU ``main``. So the import in
:func:`resolve_default_authority` genuinely fails in this environment; this
is not a simulated/mocked failure, and the refusal it produces is real,
first-class evidence, not a placeholder.
"""

from __future__ import annotations

import importlib

from .errors import ConceptAuthorityUnavailable
from .port import ConceptAuthorityPort

__all__ = ["AUTHORITY_MODULE", "resolve_default_authority"]

AUTHORITY_MODULE = "agent_utilities.governance.concept_reservation"


def resolve_default_authority() -> ConceptAuthorityPort:
    """Attempt to reach RMDD-16's native authority with no injected port.

    Always raises :class:`ConceptAuthorityUnavailable`. RMDD-16 owns
    constructing a *live* authority (it requires a connected epistemic-graph
    engine handle plus this repository's namespace/range policy) — that
    construction is explicitly out of RMDD-17's scope ("central allocator
    internals" is a listed non-goal in the lane brief). This function
    therefore never constructs one, even when the module is importable; it
    only reports precisely why no default is reachable, so a caller that
    skipped injection gets a named refusal instead of a silent local
    allocator or an ID minted without authority.
    """

    try:
        importlib.import_module(AUTHORITY_MODULE)
    except ImportError as exc:
        raise ConceptAuthorityUnavailable(
            f"could not import {AUTHORITY_MODULE} "
            "(RMDD-16's authority is not present on this agent-utilities checkout)",
            cause=exc,
        ) from exc
    raise ConceptAuthorityUnavailable(
        f"{AUTHORITY_MODULE} is importable, but RMDD-17 does not construct a "
        "live authority itself — that requires a connected epistemic-graph "
        "engine handle and a namespace policy, which is RMDD-16/RMDD-23 scope. "
        "Inject a ConceptAuthorityPort explicitly (the MCP/CLI entrypoint "
        "RMDD-20 owns is the intended caller)."
    )
