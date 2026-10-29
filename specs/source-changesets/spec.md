# RM-SCM — Git import, export and mirror sync for engine changesets

Owner: repository-manager. Cross-repo IDs: MEDULLA-SCM-R010, ARBORA-OPS-R848, ARBORA-OPS-R851, SCION-SCM-R001, MERISTEM-SCM-R001, PIPE-GATES-R006.

## Purpose and user stories

The engine holds file changesets as the source of truth (MEDULLA-SCM). The public and internal git hosts stay mirrors. repository-manager moves history between them. It imports git history into changesets. It exports each landed changeset to the mirrors. It serves the `source.*` development tools inside arbora MCP.

## Requirements and acceptance

The rows are in [requirements.md](requirements.md). RM-SCM-R001 imports history. RM-SCM-R002 keeps the mirrors in sync. RM-SCM-R003 serves the development tools. RM-SCM-R004 runs one migration wave step for one repository.

## Why a new spec

The search covered RM-MATERIALIZE, RM-IDENTITY, RM-CONNECTOR, RM-GOVERNANCE and RM-RELEASE. RM-MATERIALIZE writes one governed proposal to git. None of them imports history into engine changesets or syncs a mirror. MEDULLA-SCM-R010 owns the fidelity of one commit mapping, not the connector behavior.

## Scope and interfaces

In scope: git import, mirror export and pull-back, the served development tools and the per-repository wave step. Out of scope: changeset storage and merge (MEDULLA-SCM), served operations and admission (ARBORA-OPS-R848), the source client (SCION-SCM-R001), and the wave order (PIPE-GATES-R006).

## Traceability

| Requirement | Design section | Test ID |
|---|---|---|
| RM-SCM-R001 | plan.md, Import | T-001 |
| RM-SCM-R002 | plan.md, Mirror sync | T-002 |
| RM-SCM-R003 | plan.md, Development tools | T-003 |
| RM-SCM-R004 | plan.md, Wave step | T-004 |
