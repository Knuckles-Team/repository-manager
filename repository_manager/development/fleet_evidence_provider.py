"""Read-only, source-derived provider for fleet release evidence selectors.

The CLI/MCP reference is only a lookup key.  A trusted application composition
must configure this provider with its canonical workspace, mirrors, and frozen
plan supplier.  None of those inputs is obtained from the reference or a
caller-supplied path.  The provider does not execute a release stage.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .fleet_manifest import (
    capture_complete_manifest_census,
    reconcile_fleet_census,
    validate_fleet_manifest_mirrors,
)
from .fleet_release_gate import (
    FleetEvidenceReference,
    FleetReleaseGateError,
    FleetReleasePreflight,
    prepare_fleet_release,
)
from .workspace_release_plan import FrozenReleasePlan, validate_frozen_release_plan

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class ReadOnlyFleetEvidenceProvider:
    """Reconstruct evidence from configured source authority on every read.

    ``source_commit`` is the trusted canonical-manifest revision supplied by
    application composition.  The current workspace root is not itself a Git
    checkout, so this provider does not pretend to infer that revision from a
    repository URL or from the transport selector.
    """

    root: Path
    runtime: Path
    seed: Path
    source_commit: str
    plan_supplier: Callable[[], FrozenReleasePlan]

    def _validate_authority(
        self, reference: FleetEvidenceReference
    ) -> FrozenReleasePlan:
        if type(reference) is not FleetEvidenceReference:
            raise FleetReleaseGateError("fleet evidence reference is required")
        if type(self.source_commit) is not str or not _COMMIT.fullmatch(
            self.source_commit
        ):
            raise FleetReleaseGateError("trusted manifest revision is unavailable")
        if not all(
            isinstance(path, Path) and path.is_absolute()
            for path in (
                self.root,
                self.runtime,
                self.seed,
            )
        ):
            raise FleetReleaseGateError("trusted fleet paths must be absolute")
        if not callable(self.plan_supplier):
            raise FleetReleaseGateError("trusted frozen plan supplier is unavailable")
        plan = self.plan_supplier()
        validate_frozen_release_plan(plan)
        if plan.digest != reference.plan_digest:
            raise FleetReleaseGateError("fleet plan selector differs from authority")
        return plan

    def __call__(self, reference: FleetEvidenceReference) -> FleetReleasePreflight:
        # Keep the trusted-input checks inside the same exception boundary as
        # source reconstruction, while preserving their explicit gate errors.
        try:
            plan = self._validate_authority(reference)
            # Check mirrors before scanning any source tree.  This is read only.
            mirror = validate_fleet_manifest_mirrors(
                self.root / "workspace.yml", runtime=self.runtime, seed=self.seed
            )
            closure, census, universe = capture_complete_manifest_census(
                self.root, source_commit=self.source_commit
            )
            if mirror.source_digest != closure.manifest_sha256:
                raise FleetReleaseGateError(
                    "fleet manifest changed during evidence read"
                )
            order = reconcile_fleet_census(
                self.root, closure, census, plan.graph, universe=universe
            )
            if (
                order.digest != reference.order_digest
                or census.digest != reference.census_digest
                or universe.digest != reference.universe_digest
            ):
                raise FleetReleaseGateError(
                    "fleet evidence selector differs from sources"
                )
            return prepare_fleet_release(
                self.root,
                runtime=self.runtime,
                seed=self.seed,
                plan=plan,
                order=order,
                census=census,
                universe=universe,
            )
        except FleetReleaseGateError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise FleetReleaseGateError(
                "trusted fleet evidence is unavailable"
            ) from exc
