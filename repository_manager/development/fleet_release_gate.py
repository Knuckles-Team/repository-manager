"""Read-only fleet release preflight before any release effect.

This is the shared boundary for setup/install/bump/land/build/push adapters.
No adapter may treat the older phase plan or a caller-provided digest as a
substitute for this reverified manifest, source, graph, and stage identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .fleet_census import CensusReceipt
from .fleet_manifest import (
    FleetMirrorReceipt,
    load_fleet_manifest,
    reconcile_fleet_census,
    validate_fleet_manifest_mirrors,
)
from .fleet_source_universe import SourceUniverseReceipt, verify_source_universe
from .workspace_release import FleetOrderReceipt
from .workspace_release_plan import (
    FleetReleaseBinding,
    FrozenReleasePlan,
    StageKind,
    bind_fleet_release_plan,
)

FLEET_RELEASE_ACTIVATION = "closed_pending_source_universe_proof"


class FleetReleaseGateError(ValueError):
    """A release cannot be authorized from incomplete or drifting evidence."""


def require_legacy_release_route(operation: str) -> None:
    """Keep legacy routes usable only while the fleet cutover is closed.

    Once activation opens, an old project-map/phase route must fail before
    creating a background job or release effect. Successful fleet execution
    requires an exact stage permit through :func:`authorize_fleet_stage`.
    """

    if FLEET_RELEASE_ACTIVATION == "closed_pending_source_universe_proof":
        return
    raise FleetReleaseGateError(
        f"legacy {operation} route is closed; verified fleet stage required"
    )


@dataclass(frozen=True, slots=True)
class FleetReleasePreflight:
    root: Path
    runtime: Path
    seed: Path
    plan: FrozenReleasePlan
    order: FleetOrderReceipt
    census: CensusReceipt
    mirror: FleetMirrorReceipt
    binding: FleetReleaseBinding
    universe: SourceUniverseReceipt | None = None


@dataclass(frozen=True, slots=True)
class FleetStagePermit:
    stage_id: str
    kind: StageKind
    project_id: str
    stage_input_digest: str
    fleet_binding_digest: str


@dataclass(frozen=True, slots=True)
class FleetEvidenceReference:
    """Bounded transport selector, never authority by itself."""

    plan_digest: str
    order_digest: str
    census_digest: str
    universe_digest: str
    stage_id: str
    kind: StageKind
    project_id: str
    digest: str

    def to_json(self) -> bytes:
        payload = self._payload()
        return json.dumps(
            {**payload, "digest": self.digest}, sort_keys=True, separators=(",", ":")
        ).encode()

    def _payload(self) -> dict[str, str]:
        return {
            "schema": "fleet_evidence_reference/v1",
            "plan_digest": self.plan_digest,
            "order_digest": self.order_digest,
            "census_digest": self.census_digest,
            "universe_digest": self.universe_digest,
            "stage_id": self.stage_id,
            "kind": self.kind.value,
            "project_id": self.project_id,
        }


def _reference_digest(payload: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_REFERENCE_FIELDS = frozenset(
    (
        "schema",
        "plan_digest",
        "order_digest",
        "census_digest",
        "universe_digest",
        "stage_id",
        "kind",
        "project_id",
        "digest",
    )
)
_REFERENCE_DIGEST_FIELDS = (
    "plan_digest",
    "order_digest",
    "census_digest",
    "universe_digest",
    "digest",
)


def _unique_reference_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = dict(pairs)
    if len(document) != len(pairs):
        raise ValueError("duplicate envelope key")
    return document


def _validated_reference_document(raw: bytes) -> dict[str, str]:
    document = json.loads(raw, object_pairs_hook=_unique_reference_pairs)
    if (
        type(document) is not dict
        or set(document) != _REFERENCE_FIELDS
        or document["schema"] != "fleet_evidence_reference/v1"
    ):
        raise ValueError("invalid envelope shape")
    if any(type(value) is not str or len(value) > 256 for value in document.values()):
        raise ValueError("invalid envelope field")
    _validate_reference_digests(document)
    return document


def _validate_reference_digests(document: dict[str, str]) -> None:
    for key in _REFERENCE_DIGEST_FIELDS:
        value = document[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("invalid envelope digest")
    payload = {key: value for key, value in document.items() if key != "digest"}
    if _reference_digest(payload) != document["digest"]:
        raise ValueError("envelope digest differs")


def parse_fleet_evidence_reference(raw: bytes) -> FleetEvidenceReference:
    """Parse an untrusted CLI/MCP selector with exact keys and bounded size."""

    if type(raw) is not bytes or len(raw) > 4096:
        raise FleetReleaseGateError("fleet evidence reference exceeds bound")
    try:
        document = _validated_reference_document(raw)
        return FleetEvidenceReference(
            document["plan_digest"],
            document["order_digest"],
            document["census_digest"],
            document["universe_digest"],
            document["stage_id"],
            StageKind(document["kind"]),
            document["project_id"],
            document["digest"],
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        KeyError,
        ValueError,
    ) as exc:
        raise FleetReleaseGateError("fleet evidence reference is invalid") from exc


def reference_for_fleet_stage(
    preflight: FleetReleasePreflight, *, stage_id: str, kind: StageKind, project_id: str
) -> FleetEvidenceReference:
    """Mint a selector only after exact stage authorization and re-preflight."""

    permit = authorize_fleet_stage(
        preflight, stage_id=stage_id, kind=kind, project_id=project_id
    )
    return _reference_from_permit(preflight, permit)


def _reference_from_permit(
    preflight: FleetReleasePreflight, permit: FleetStagePermit
) -> FleetEvidenceReference:
    if preflight.universe is None:
        raise FleetReleaseGateError("fleet source universe proof is required")
    payload = {
        "schema": "fleet_evidence_reference/v1",
        "plan_digest": preflight.plan.digest,
        "order_digest": preflight.order.digest,
        "census_digest": preflight.census.digest,
        "universe_digest": preflight.universe.digest,
        "stage_id": permit.stage_id,
        "kind": permit.kind.value,
        "project_id": permit.project_id,
    }
    return FleetEvidenceReference(
        payload["plan_digest"],
        payload["order_digest"],
        payload["census_digest"],
        payload["universe_digest"],
        permit.stage_id,
        permit.kind,
        permit.project_id,
        _reference_digest(payload),
    )


def authorize_fleet_evidence_reference(
    preflight: FleetReleasePreflight, reference: FleetEvidenceReference
) -> FleetStagePermit:
    """Resolve transport identity against trusted preflight, then reverify it."""

    if type(reference) is not FleetEvidenceReference:
        raise FleetReleaseGateError("fleet evidence reference is required")
    permit = authorize_fleet_stage(
        preflight,
        stage_id=reference.stage_id,
        kind=reference.kind,
        project_id=reference.project_id,
    )
    expected = _reference_from_permit(preflight, permit)
    if expected != reference:
        raise FleetReleaseGateError("fleet evidence reference differs from preflight")
    return permit


def prepare_fleet_release(
    root: Path,
    *,
    runtime: Path,
    seed: Path,
    plan: FrozenReleasePlan,
    order: FleetOrderReceipt,
    census: CensusReceipt,
    universe: SourceUniverseReceipt | None = None,
) -> FleetReleasePreflight:
    """Rebuild exact inputs and refuse an unproven source universe.

    The ordinary census producer emits ``complete=False``. A complete census
    must carry a reverified independent Git-tree proof. Fleet activation stays
    closed until canonical metadata and live consumers can cut over together.
    This function never performs a release effect.
    """

    if FLEET_RELEASE_ACTIVATION != "open":
        raise FleetReleaseGateError("fleet release activation is closed")
    if type(census) is not CensusReceipt or census.complete is not True:
        raise FleetReleaseGateError("fleet source universe is not complete")
    if type(universe) is not SourceUniverseReceipt:
        raise FleetReleaseGateError(
            "independent fleet source universe proof is required"
        )
    try:
        mirror = validate_fleet_manifest_mirrors(
            root / "workspace.yml", runtime=runtime, seed=seed
        )
        closure = load_fleet_manifest(root)
        if not verify_source_universe(root, closure, universe):
            raise FleetReleaseGateError("fleet source universe proof failed")
        derived = reconcile_fleet_census(
            root, closure, census, plan.graph, universe=universe
        )
        if derived != order:
            raise FleetReleaseGateError("fleet order differs from verified sources")
        binding = bind_fleet_release_plan(plan, order)
    except FleetReleaseGateError:
        raise
    except (OSError, ValueError) as exc:
        raise FleetReleaseGateError("fleet release preflight failed") from exc
    if mirror.source_digest != order.manifest_digest:
        raise FleetReleaseGateError("fleet mirror identity differs from order")
    return FleetReleasePreflight(
        root, runtime, seed, plan, order, census, mirror, binding, universe
    )


def authorize_fleet_stage(
    preflight: FleetReleasePreflight,
    *,
    stage_id: str,
    kind: StageKind,
    project_id: str,
) -> FleetStagePermit:
    """Issue an exact stage token only for a preflighted plan declaration."""

    if type(preflight) is not FleetReleasePreflight:
        raise FleetReleaseGateError("fleet preflight is required")
    current = prepare_fleet_release(
        preflight.root,
        runtime=preflight.runtime,
        seed=preflight.seed,
        plan=preflight.plan,
        order=preflight.order,
        census=preflight.census,
        universe=preflight.universe,
    )
    if current != preflight:
        raise FleetReleaseGateError("fleet release preflight has drifted")
    matches = tuple(
        stage
        for stage in preflight.plan.stages
        if stage.stage_id == stage_id
        and stage.kind is kind
        and stage.project_id == project_id
    )
    if len(matches) != 1:
        raise FleetReleaseGateError("fleet stage is not in the frozen release plan")
    stage = matches[0]
    return FleetStagePermit(
        stage.stage_id,
        stage.kind,
        stage.project_id,
        stage.input_digest,
        preflight.binding.digest,
    )
