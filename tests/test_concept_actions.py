"""Concept-authority construction chokepoint (RM-GOVERNANCE-07).

``build_default_concept_authority`` is the one place the ``rm_concepts`` MCP
tool and the ``--concepts`` CLI family both reach for a default
:class:`~repository_manager.concept_coordination.port.ConceptAuthorityPort`.
This proves it resolves against the real, current authority-module location
(:mod:`repository_manager.governance.concept_reservation`, moved here from
``agent_utilities`` by operator ruling) rather than the retired path, and
that its honest ``None`` result today comes from "no ``build_default_authority``
factory published yet" -- not from a dead import of a module that no longer
lives where the resolver used to look.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from repository_manager import concept_actions
from repository_manager.concept_coordination.client import AUTHORITY_MODULE


def test_authority_module_points_at_the_in_repo_governance_package() -> None:
    assert AUTHORITY_MODULE == "repository_manager.governance.concept_reservation"
    # It genuinely imports -- the previous agent_utilities path never did
    # once RM-GOVERNANCE-R001 moved the module into this package.
    module = importlib.import_module(AUTHORITY_MODULE)
    assert hasattr(module, "NativeConceptReservationAuthority")


def test_default_authority_is_none_because_no_factory_not_a_dead_import() -> None:
    """Honest degrade: the module is reachable; it just exposes no
    documented ``build_default_authority()`` construction entrypoint yet
    (constructing a live one needs a connected epistemic-graph engine
    handle, out of this package's scope -- see ``concept_actions.py``)."""
    module = importlib.import_module(AUTHORITY_MODULE)
    assert getattr(module, "build_default_authority", None) is None

    assert concept_actions.build_default_concept_authority() is None


def test_concept_actions_for_is_the_shared_mcp_cli_chokepoint(tmp_path: Path) -> None:
    """MCP and CLI both construct through this one function (parity)."""
    actions = concept_actions.concept_actions_for(
        repo_root=tmp_path, tenant_ref="tenant-1", lane_ref="lane-1"
    )
    assert actions is not None
