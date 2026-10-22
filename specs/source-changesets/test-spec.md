# RM-SCM — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Expected observation |
|---|---|---|---|
| T-R001.1 | RM-SCM-R001.1 | Unit | A unit test round-trips both models. |
| T-R001.2 | RM-SCM-R001.2 | Unit | A test imports a 50-commit fixture twice and asserts 50 changesets, then 0. |
| T-R001.3 | RM-SCM-R001.3 | Unit | A served test calls `rm_source_import` and reads the receipt. |
| T-R001.4 | RM-SCM-R001.4 | Unit | A test asserts each code and no partial changeset. |
| T-R001.5 | RM-SCM-R001.5 | Unit | A test reads the receipt fields. |
| T-R001.6 | RM-SCM-R001.6 | Served | `tests/source_changesets/test_import_served.py::test_import_full_history` passes. |
| T-R002.1 | RM-SCM-R002.1 | Unit | A unit test round-trips both models. |
| T-R002.2 | RM-SCM-R002.2 | Unit | A test lands one changeset and pulls one mirror commit back. |
| T-R002.3 | RM-SCM-R002.3 | Unit | A served test lands through arbora and observes the mirror push. |
| T-R002.4 | RM-SCM-R002.4 | Unit | A test diverges a fixture mirror and asserts the code and no push. |
| T-R002.5 | RM-SCM-R002.5 | Unit | A test reads the receipts. |
| T-R002.6 | RM-SCM-R002.6 | Served | `tests/source_changesets/test_mirror_sync_served.py::test_land_then_mirror` passes. |
| T-R003.1 | RM-SCM-R003.1 | Unit | A test lists the eight tools and their hints. |
| T-R003.2 | RM-SCM-R003.2 | Unit | A test asserts a passed-through `CONFLICT` with paths. |
| T-R003.3 | RM-SCM-R003.3 | Served | `tests/source_changesets/test_dev_tools_served.py::test_commit_and_land_through_arbora` passes. |
| T-R004.1 | RM-SCM-R004.1 | Unit | A test runs the step and the reverse step on a fixture repository. |
| T-R004.2 | RM-SCM-R004.2 | Unit | A test asserts the refusal for an unmet criterion. |
| T-R004.3 | RM-SCM-R004.3 | Unit | `tests/source_changesets/test_wave_step.py::test_step_and_reverse_receipts` passes. |
