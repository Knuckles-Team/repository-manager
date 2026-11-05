# RM-SCM — Test specification

Governing spec: [spec.md](spec.md).

Proof level: unit tests use temporary Git repositories; served tests enter through the MCP client against a recorded engine and a local bare repository standing for a mirror. No test is built; every test below is planned.

## Planned

- `tests/source_changesets/test_models.py::test_import_models_round_trip` (planned). It round-trips both models. *(RM-SCM-R001.1)*
- `tests/source_changesets/test_import.py::test_import_is_incremental` (planned). It imports a 50-commit fixture twice and asserts 50 changesets, then 0. *(RM-SCM-R001.2)*
- `tests/source_changesets/test_import_served.py::test_rm_source_import_returns_receipt` (planned). It calls `rm_source_import` through the MCP client and reads the receipt. *(RM-SCM-R001.3)*
- `tests/source_changesets/test_import.py::test_refusals_leave_no_partial_changeset` (planned). It asserts each code and no partial changeset. *(RM-SCM-R001.4)*
- `tests/source_changesets/test_import.py::test_receipt_fields` (planned). It reads the receipt fields. *(RM-SCM-R001.5)*
- `tests/source_changesets/test_import_served.py::test_import_full_history` (planned). It asserts the time and memory bounds. *(RM-SCM-R001.6)*
- `tests/source_changesets/test_models.py::test_mirror_models_round_trip` (planned). It round-trips both models. *(RM-SCM-R002.1)*
- `tests/source_changesets/test_mirror.py::test_land_pushes_and_mirror_commit_imports_back` (planned). It lands one changeset and pulls one mirror commit back. *(RM-SCM-R002.2)*
- `tests/source_changesets/test_mirror_sync_served.py::test_land_through_arbora_observes_mirror_push` (planned). It lands through arbora and observes the mirror push. *(RM-SCM-R002.3)*
- `tests/source_changesets/test_mirror.py::test_diverged_mirror_refuses_without_push` (planned). It diverges a fixture mirror and asserts the code and no push. *(RM-SCM-R002.4)*
- `tests/source_changesets/test_mirror.py::test_receipts_per_host` (planned). It reads the receipts. *(RM-SCM-R002.5)*
- `tests/source_changesets/test_mirror_sync_served.py::test_land_then_mirror` (planned). It asserts the 60 s bound and the equal tree hash. *(RM-SCM-R002.6)*
- `tests/source_changesets/test_dev_tools.py::test_eight_tools_and_their_hints` (planned). It lists the eight tools and their hints. *(RM-SCM-R003.1)*
- `tests/source_changesets/test_dev_tools.py::test_conflict_refusal_passes_through_with_paths` (planned). It asserts a passed-through `conflict` with paths. *(RM-SCM-R003.2)*
- `tests/source_changesets/test_dev_tools_served.py::test_commit_and_land_through_arbora` (planned). It commits and lands one file and reads it back. *(RM-SCM-R003.3)*
- `tests/source_changesets/test_wave_step.py::test_step_and_reverse_on_a_fixture_repository` (planned). It runs the step and the reverse step. *(RM-SCM-R004.1)*
- `tests/source_changesets/test_wave_step.py::test_unmet_entry_criterion_refuses` (planned). It asserts the refusal for an unmet criterion. *(RM-SCM-R004.2)*
- `tests/source_changesets/test_wave_step.py::test_step_and_reverse_receipts` (planned). It asserts both receipts. *(RM-SCM-R004.3)*

## Quality

Run `uv run pytest tests/source_changesets`, the repository pre-commit, Ruff, mypy and CCCC (no new function over cyclomatic 10 or cognitive 15). jscpd and dupehound are not configured repository hooks; run them where available and report a missing tool. No second Git runner and no second engine call is added.
