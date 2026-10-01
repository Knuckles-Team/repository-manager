"""Development governance: lane arbitration and concept-ID coordination.

Moved here from ``agent_utilities.governance`` by operator ruling:
repository-manager is the development-governance tool, so it owns

* :mod:`.lanes` — the shared-resource arbitration classes (partition, lease,
  append-only fragments, guarded tree mutation) every lane on a host shares;
* :mod:`.concept_hierarchy` / :mod:`.concept_lineage` — the OKF-CIS concept-id
  grammar, the closed domain vocabulary and slug registry, and the lineage
  rules a repository's concept-governance gate enforces;
* :mod:`.concept_allocator` — same-host concept-id reservation over per-lane
  append-only ledger fragments;
* :mod:`.concept_reservation` — the separate-host authority port, which fails
  closed when the native graph authority is unavailable;
* :mod:`.lane_guard` — the fleet lane-guard pre-commit gate;
* :mod:`.promotion` — how far the deployed ref lags ``main`` (merge is not deploy);
* :mod:`.cli` — the ``repository-manager-governance`` command surface.

The merge queue that also lived in ``agent_utilities.governance`` was not
moved: :mod:`repository_manager.merge_queue` is its generalized successor, and a
repository opts in with a root ``.mergequeue.yaml``.
"""
