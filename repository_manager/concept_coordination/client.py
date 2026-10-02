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

**Where the authority module actually lives now (RM-GOVERNANCE-07).** RMDD-16's
authority (``ConceptReservationService``, ``NativeConceptReservationAuthority``,
``FixtureConceptReservationAuthority``, states
reserved/materialized/landed/released/expired/tombstoned, fenced transitions,
``reconcile_projection``) was never an ancestor of agent-utilities ``main`` as
``agent_utilities/governance/concept_reservation.py`` — it has since moved, by
operator ruling, into this repository itself as
:mod:`repository_manager.governance.concept_reservation` (repository-manager
is the development-governance tool; see that package's ``__init__`` for the
full migration note). So the module genuinely IS importable today, in this
same package. What it still does not expose is a documented
``build_default_authority()`` construction entrypoint:
:class:`~repository_manager.governance.concept_reservation.NativeConceptReservationAuthority`
needs a *connected* epistemic-graph engine handle
(``NativeConceptReservationPort``) injected into its constructor, and nothing
in this codebase establishes a standard way to reach a live one from here —
that remains RMDD-16/RMDD-23 scope, not RMDD-17's ("central allocator
internals" is a listed non-goal in the lane brief). :func:`resolve_default_authority`
therefore still always refuses, but for the real, actionable reason (no live
engine injected) rather than a now-permanently-false one (the module missing
from an ``agent_utilities`` checkout it was deliberately moved out of).
"""

from __future__ import annotations

import importlib

from .errors import ConceptAuthorityUnavailable
from .port import ConceptAuthorityPort

__all__ = ["AUTHORITY_MODULE", "resolve_default_authority"]

AUTHORITY_MODULE = "repository_manager.governance.concept_reservation"


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
            "(RMDD-16's authority module is not present in this checkout)",
            cause=exc,
        ) from exc
    raise ConceptAuthorityUnavailable(
        f"{AUTHORITY_MODULE} is importable, but RMDD-17 does not construct a "
        "live authority itself — that requires a connected epistemic-graph "
        "engine handle and a namespace policy, which is RMDD-16/RMDD-23 scope. "
        "Inject a ConceptAuthorityPort explicitly (the MCP/CLI entrypoint "
        "RMDD-20 owns is the intended caller)."
    )
