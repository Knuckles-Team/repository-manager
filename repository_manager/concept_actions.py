"""RMDD-20: the one MCP/CLI-shared concept-coordination application service.

This module is the chokepoint both the ``rm_concepts`` MCP tool
(:mod:`repository_manager.mcp_tools.concepts`) and the ``--concepts`` CLI
family (:mod:`repository_manager.cli_commands.concepts`) call — neither
adapter constructs a :class:`~repository_manager.concept_coordination.action_core.ConceptCoordinationActions`
or an authority of its own, so MCP and CLI can never diverge in behavior
(the lane's parity requirement).

``build_default_concept_authority`` is the "entrypoint that constructs the
real RMDD-16 authority and injects it into ``ConceptCoordinationActions``"
the RMDD-17 lane brief asked RMDD-20 to write. It is a best-effort attempt,
never a fabrication: RMDD-16's authority module
(``agent_utilities.governance.concept_reservation``) is not an ancestor of
agent-utilities ``main`` as of this lane (verified 2026-08-10; see
``repository_manager/concept_coordination/client.py`` for the full account
of exactly which commit and which branches carry it), so today this always
returns ``None`` in this environment and every caller falls through to
``ConceptCoordinationActions``'s own ``resolve_default_authority()`` — which
raises a named ``ConceptAuthorityUnavailable`` preserving the real
``ImportError`` as ``__cause__`` (H-12: never discard an exception cause).
No fixture, in-memory allocator, or fabricated authority is ever substituted
for the real one in this module.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from repository_manager.concept_coordination import (
    AUTHORITY_MODULE,
    ConceptAuthorityPort,
    ConceptClaimRecord,
    ConceptClaimRequest,
    ConceptClaimState,
    ConceptCoordinationActions,
    ConceptCoordinationError,
)

__all__ = [
    "CONCEPT_ACTIONS",
    "build_default_concept_authority",
    "concept_actions_for",
    "dispatch",
]

CONCEPT_ACTIONS: tuple[str, ...] = (
    "reserve",
    "list",
    "get",
    "release",
    "materialize",
    "verify_candidate",
    "verify_generation",
    "reconcile",
)


def build_default_concept_authority() -> ConceptAuthorityPort | None:
    """Best-effort construction of RMDD-16's real, live concept authority.

    Returns ``None`` whenever the authority module is absent, or is
    importable but exposes no documented live-construction entrypoint, or
    that construction raises for any reason — it never fabricates a
    fixture/local allocator as a substitute (repository-manager AGENTS.md
    "optional dependency" guardrail: ``try/except ImportError`` only, no
    module-level import of ``agent_utilities.governance.concept_reservation``
    anywhere in this module). ``None`` is a legitimate, honest result: the
    caller (``concept_actions_for``) passes it straight through to
    ``ConceptCoordinationActions``, whose own ``resolve_default_authority()``
    re-attempts the same import and raises the named refusal.
    """

    try:
        import importlib

        module = importlib.import_module(AUTHORITY_MODULE)
    except ImportError:
        return None
    factory = getattr(module, "build_default_authority", None)
    if factory is None:
        return None
    try:
        authority = factory()
    except Exception:  # noqa: BLE001 - construction failure means "unavailable"
        return None
    if not isinstance(authority, ConceptAuthorityPort):
        return None
    return authority


def concept_actions_for(
    *, repo_root: str | Path, tenant_ref: str, lane_ref: str
) -> ConceptCoordinationActions:
    """The one construction path MCP and CLI both call (H-11 chokepoint)."""

    return ConceptCoordinationActions(
        repo_root=Path(repo_root),
        tenant_ref=tenant_ref,
        lane_ref=lane_ref,
        authority=build_default_concept_authority(),
    )


def _default_window(
    kwargs: dict[str, Any], *, ttl_seconds: int = 7 * 24 * 3600
) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    created_at = kwargs.get("created_at") or now
    expires_at = kwargs.get("expires_at") or (now + timedelta(seconds=ttl_seconds))
    return created_at, expires_at


def _record_payload(record: ConceptClaimRecord) -> dict[str, Any]:
    return record.canonical_payload()


def dispatch(
    action: str,
    *,
    repo_root: str,
    tenant_ref: str,
    lane_ref: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resolve and execute one concept action; MCP and CLI share this exactly.

    Returns ``{"ok": True, ...}`` on success or ``{"ok": False, "refused":
    ..., "error_code": ...}`` on a named refusal (C-10) — a
    :class:`~repository_manager.concept_coordination.errors.ConceptCoordinationError`
    is never allowed to propagate as a bare traceback to either adapter.
    """

    from agent_utilities.mcp.action_dispatch import resolve_action

    resolved = resolve_action(
        action, CONCEPT_ACTIONS, service="repository-manager-concepts"
    )
    if isinstance(resolved, dict):
        return resolved

    actions = concept_actions_for(
        repo_root=repo_root, tenant_ref=tenant_ref, lane_ref=lane_ref
    )
    try:
        result = _dispatch_resolved(actions, resolved, kwargs)
    except ConceptCoordinationError as exc:
        return {
            "ok": False,
            "refused": str(exc),
            "error_code": exc.code.value,
            "unreachable": getattr(exc, "unreachable", None),
        }
    except (KeyError, LookupError, TypeError, ValueError) as exc:
        # A missing/malformed request field (e.g. a required kwarg the
        # caller omitted, or a value that fails the typed contract's own
        # validation) is an invalid request, not a crash — named per C-10
        # rather than left to propagate as a bare traceback (H-12: the
        # original exception is still visible via `str(exc)`, never
        # discarded).
        return {
            "ok": False,
            "refused": f"invalid concept request: {exc}",
            "error_code": "invalid_request",
        }
    return {"ok": True, **result}


def _dispatch_resolved(
    actions: ConceptCoordinationActions, resolved: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    if resolved == "reserve":
        created_at, expires_at = _default_window(kwargs)
        request = ConceptClaimRequest(
            tenant_ref=kwargs.get("tenant_ref") or actions.tenant_ref,
            concept_id=kwargs["concept_id"],
            namespace=kwargs["namespace"],
            repository_ref=kwargs["repository_ref"],
            lane_ref=kwargs.get("lane_ref") or actions.lane_ref,
            owner_ref=kwargs["owner_ref"],
            request_key_ref=kwargs["request_key_ref"],
            purpose=kwargs["purpose"],
            design_ref=kwargs.get("design_ref"),
            branch=kwargs.get("branch"),
            base_sha=kwargs.get("base_sha"),
            workitem_ref=kwargs.get("workitem_ref"),
            run_trace_ref=kwargs.get("run_trace_ref"),
            created_at=created_at,
            expires_at=expires_at,
            provenance_refs=tuple(kwargs.get("provenance_refs") or ()),
        )
        return {"record": _record_payload(actions.reserve(request))}

    if resolved == "get":
        return {"record": _record_payload(actions.get(kwargs["reservation_id"]))}

    if resolved == "list":
        state = ConceptClaimState(kwargs["state"]) if kwargs.get("state") else None
        rows, cursor = actions.list(
            namespace=kwargs.get("namespace"),
            state=state,
            concept_prefix=kwargs.get("concept_prefix"),
            limit=kwargs.get("limit", 1000),
            cursor=kwargs.get("cursor"),
        )
        return {
            "records": [_record_payload(row) for row in rows],
            "cursor": cursor,
        }

    if resolved == "release":
        record = actions.release(
            kwargs["reservation_id"],
            owner_ref=kwargs["owner_ref"],
            expected_fence=kwargs["expected_fence"],
        )
        return {"record": _record_payload(record)}

    if resolved == "materialize":
        record = actions.materialize(
            kwargs["reservation_id"],
            owner_ref=kwargs["owner_ref"],
            expected_fence=kwargs["expected_fence"],
        )
        return {"record": _record_payload(record)}

    if resolved == "verify_candidate":
        from repository_manager.development import Candidate

        candidate = Candidate.model_validate(kwargs["candidate"])
        outcome = actions.verify_candidate(
            candidate, repo_path=Path(kwargs["repo_path"])
        )
        return {
            "verification": dataclasses.asdict(outcome) | {"passed": outcome.passed}
        }

    if resolved == "verify_generation":
        from repository_manager.development import Candidate, Generation

        generation = Generation.model_validate(kwargs["generation"])
        candidates = tuple(
            Candidate.model_validate(item) for item in kwargs["candidates"]
        )
        generation_verification = actions.verify_generation(
            generation, candidates, repo_path=Path(kwargs["repo_path"])
        )
        return {
            "verification": {
                "generation_id": generation_verification.generation_id,
                "passed": generation_verification.passed,
                "per_candidate": {
                    candidate_id: dataclasses.asdict(item) | {"passed": item.passed}
                    for candidate_id, item in generation_verification.per_candidate.items()
                },
                "collisions": {
                    key: list(value)
                    for key, value in generation_verification.collisions.items()
                },
            }
        }

    if resolved == "reconcile":
        report = actions.reconcile(
            source_tree_ish=kwargs.get("source_tree_ish", "HEAD")
        )
        return {"report": report.to_dict()}

    raise AssertionError(f"unhandled resolved concept action {resolved!r}")
