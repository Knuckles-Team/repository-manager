# CONN-REPOSITORY-MANAGER — Git import, mirror push and source cutover

Owner: repository-manager. Cross-repo IDs: EG-SCM-R010, EG-SCM-R012, ARBORA-OPS-R113, SDK-REPOSITORY-TRANSPORT-R003.

## Purpose and user stories

The engine becomes the source of truth for one repository at a time (EG-SCM-R012).
This connector moves git objects between the engine and each git mirror over the git wire protocol.

1. **Repository operator (P1):** the operator imports a repository's full history into engine changesets.
2. **Mirror reader (P1):** each git mirror shows every accepted changeset within 60 seconds.
3. **Operator on failure (P1):** a rollback makes the mirror the write target again in 10 minutes or less.

## Requirements and acceptance

Every requirement ID lives in [requirements.md](requirements.md): `CONN-REPOSITORY-MANAGER-R001`–`R004`.

## Why a new spec

The search covered `git grep -n -iE "mirror|import|export|push|remote" -- specs`.
It read RM-MATERIALIZE, RM-IDENTITY, RM-CONNECTOR, RM-GOVERNANCE and RM-RELEASE.
RM-MATERIALIZE turns an approved proposal into one commit. RM-IDENTITY rewrites history once with approval.
No spec owns a continuous mirror of engine changesets or the import of a whole history into the engine.

## Scope and interfaces

Owned: history import, the mirror push, external-commit import, drift detection and the rollback switch.
Not owned: changeset storage and git object IDs (EG-SCM), the read bridge (ARBORA-OPS-R113).

## Traceability

| Requirement | Design section | Test |
|---|---|---|
| CONN-REPOSITORY-MANAGER-R001 | plan.md "Import" | `tests/test_source_mirror_import.py` |
| CONN-REPOSITORY-MANAGER-R002 | plan.md "Mirror push" | `tests/test_source_mirror_push.py` |
| CONN-REPOSITORY-MANAGER-R003 | plan.md "Drift and external commits" | `tests/test_source_mirror_drift.py` |
| CONN-REPOSITORY-MANAGER-R004 | plan.md "Local merge queue" | `tests/test_source_mirror_queue_retire.py` |
