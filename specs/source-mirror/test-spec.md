# CONN-REPOSITORY-MANAGER test contract

| Row | Test | Observation |
|---|---|---|
| `CONN-REPOSITORY-MANAGER-R001` | `tests/test_source_mirror_import.py::test_import_is_topological_idempotent_and_id_preserving` | Identical commit IDs. A re-run submits nothing. |
| `CONN-REPOSITORY-MANAGER-R002` | `tests/test_source_mirror_push.py::test_push_is_leased_per_remote_and_never_reports_partial_success` | One receipt per remote. A partial push is never success. |
| `CONN-REPOSITORY-MANAGER-R003` | `tests/test_source_mirror_drift.py::test_external_commit_imports_and_drift_is_reported` | External commits import. Drift past 5 minutes is a finding. |
| `CONN-REPOSITORY-MANAGER-R004` | `tests/test_source_mirror_queue_retire.py::test_queue_refuses_engine_primary_and_rollback_restores_it` | The queue refuses `engine_primary`. Rollback restores it. |

Tests use disposable git repositories and a fake arbora endpoint. They contact no remote.
