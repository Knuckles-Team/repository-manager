# RM-SCM — Git import, export and mirror sync for engine changesets

Component: Source control

Owner: repository-manager. Cross-repo IDs: `MEDULLA-SCM-R010`, `ARBORA-OPS-R848`, `ARBORA-OPS-R851`, `SCION-SCM-R001`, `MERISTEM-SCM-R001`, `PIPE-GATES-R006`.

## Outcome

The engine holds file changesets as the source of truth (`MEDULLA-SCM-R010`). The public and internal git hosts stay mirrors. repository-manager moves history between them: it imports git history into changesets, exports each landed changeset to the mirrors, serves the `source.*` development tools inside arbora MCP, and runs one migration wave step per repository.

## Scope

Owner: `repository-manager`. The code is a planned package `repository_manager/source_changesets/` (`models.py`, `importer.py`, `mirror.py`, `wave.py`) and a planned tool registrar `repository_manager/mcp_tools/source.py`; none of it exists yet. It reuses `Git.git_action` (`repository_manager/repository_manager.py:2462`) for every Git process and the SDK source client for every engine effect. In scope: git import, mirror export and pull-back, the served development tools and the per-repository wave step. Out of scope: changeset storage and merge (`MEDULLA-SCM-R010`), served operations and admission (`ARBORA-OPS-R848`), the source client (`SCION-SCM-R001`) and the wave order (`PIPE-GATES-R006`).

Why a spec: the search covered `RM-MATERIALIZE`, `RM-IDENTITY`, `RM-CONNECTOR`, `RM-GOVERNANCE` and `RM-RELEASE`. `RM-MATERIALIZE` writes one governed proposal to Git. None of them imports history into engine changesets or syncs a mirror. `MEDULLA-SCM-R010` owns the fidelity of one commit mapping, not the connector behavior.

## User Scenarios & Testing

### User Story 1 - Import a repository's history (Priority: P1)

The user is a maintainer moving a repository to engine changesets. The outcome is one changeset per commit with equal tree hashes and a receipt.

**Acceptance scenarios**

1. **Given** a repository with 50 commits, **When** the maintainer imports it twice, **Then** 50 changesets exist after the first run and none are added by the second.
2. **Given** a changeset whose tree hash differs from its commit, **When** the import runs, **Then** it stops with `IMPORT_TREE_MISMATCH` and keeps the prior changesets.
3. **Given** a shallow clone, **When** the import runs, **Then** it refuses with `IMPORT_SHALLOW_HISTORY`.

Independent test: the bound tests of `RM-SCM-R001.1` to `RM-SCM-R001.6`.

### User Story 2 - Keep the git mirrors in sync (Priority: P1)

The user is a contributor who clones a public mirror. The outcome is each landed changeset visible on every mirror with an equal tree hash, and a mirror commit coming back as a changeset.

**Acceptance scenarios**

1. **Given** a landed changeset, **When** the sync runs, **Then** each mirror holds the commit within 60 s with the changeset tree hash.
2. **Given** a mirror holding a diverged commit, **When** the sync runs, **Then** it stops with `MIRROR_DIVERGED` and pushes nothing.

Independent test: the bound tests of `RM-SCM-R002.1` to `RM-SCM-R002.6`.

### User Story 3 - Edit and land through served tools (Priority: P2)

The user is an agent. The outcome is one tool per `source.*` operation with typed refusals.

**Acceptance scenarios**

1. **Given** the connector is registered with arbora, **When** the agent lists tools, **Then** the eight `rm_source_*` tools appear with read-only or destructive hints.
2. **Given** a conflicting edit, **When** the agent lands it, **Then** the `conflict` refusal names the conflicting paths.

Independent test: the bound tests of `RM-SCM-R003.1` to `RM-SCM-R003.3`.

### User Story 4 - Move one repository to engine-primary (Priority: P2)

The user is the migration operator. The outcome is a wave step, and its reverse, with a receipt.

**Acceptance scenarios**

1. **Given** the wave entry criteria hold, **When** the step runs, **Then** the repository is recorded engine-primary and the mirror sync is on.
2. **Given** an unmet criterion, **When** the step runs, **Then** it refuses with `WAVE_ENTRY_UNMET`.

Independent test: the bound tests of `RM-SCM-R004.1` to `RM-SCM-R004.3`.

## Requirements

- **FR-001**: Import maps one commit to one changeset with an equal tree hash (`RM-SCM-R001`).
- **FR-002**: Landed changesets reach every mirror and mirror commits return (`RM-SCM-R002`).
- **FR-003**: One served tool exists per `source.*` operation (`RM-SCM-R003`).
- **FR-004**: One wave step moves a repository between primaries (`RM-SCM-R004`).

## Success Criteria

- **SC-001**: A 1,000-commit import finishes under 120 s and 512 MiB (`RM-SCM-R001.6`).
- **SC-002**: A landed changeset appears on a mirror within 60 s with an equal tree hash (`RM-SCM-R002.6`).

## Edge Cases

- A re-run after a partial import resumes at the first commit without a changeset.
- A commit on a mirror whose tree hash differs from its changeset is refused by the import.
- A landing made by a client other than `rm_source_land` reaches the sync only through the mechanism chosen by the owner.

## Assumptions

- The engine is the source of truth and the git hosts are byte-identical mirrors in every wave.
- The caller supplies the engine client; the connector never holds engine credentials.
