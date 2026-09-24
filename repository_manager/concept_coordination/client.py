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

**Why this always refuses.** The authority module
(:mod:`repository_manager.governance.concept_reservation`, moved here from
agent-utilities by OQ-3) is part of this package and always importable, but a
*live* authority needs a connected epistemic-graph engine handle and the
repository's namespace/range policy — construction this module deliberately
does not perform. So the refusal is the honest answer: inject a port.
"""

from __future__ import annotations

from .errors import ConceptAuthorityUnavailable
from .port import ConceptAuthorityPort

__all__ = ["AUTHORITY_MODULE", "resolve_default_authority"]

AUTHORITY_MODULE = "repository_manager.governance.concept_reservation"


def resolve_default_authority() -> ConceptAuthorityPort:
    """Refuse, by name, when no concept authority was injected.

    Always raises :class:`ConceptAuthorityUnavailable`. RMDD-16 owns
    constructing a *live* authority (it requires a connected epistemic-graph
    engine handle plus this repository's namespace/range policy) — that
    construction is explicitly out of RMDD-17's scope ("central allocator
    internals" is a listed non-goal in the lane brief). This function
    therefore never constructs one; it only reports precisely why no default
    is reachable, so a caller that skipped injection gets a named refusal
    instead of a silent local allocator or an ID minted without authority.
    """
    raise ConceptAuthorityUnavailable(
        f"{AUTHORITY_MODULE} is importable, but RMDD-17 does not construct a "
        "live authority itself — that requires a connected epistemic-graph "
        "engine handle and a namespace policy, which is RMDD-16/RMDD-23 scope. "
        "Inject a ConceptAuthorityPort explicitly (the MCP/CLI entrypoint "
        "RMDD-20 owns is the intended caller)."
    )
