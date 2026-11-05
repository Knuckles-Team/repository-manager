# RM-RELEASE — Dependency aware fleet release

Component: Fleet release

**Owner:** repository-manager. See [requirements.md](requirements.md) for the definition of every requirement ID.

## Outcome

A maintainer can release a selected set of repositories in an order derived from declared and observed dependency edges. A contributor can reproduce the planner with fixtures in a fresh checkout, without access to any operator infrastructure. The release refuses incomplete evidence, cycles, stale inputs and partial publication before mutating a repository.

## Scope

Owner: `repository-manager`, the sole owner of census, ordering, Git execution and receipts. The code is `repository_manager/development/` (`fleet_census.py`, `fleet_manifest.py`, `fleet_source_universe.py`, `workspace_release.py`, `workspace_release_plan.py`, `fleet_release_gate.py`, `fleet_evidence_provider.py`), `repository_manager/dependency_readiness.py` and `repository_manager/rollout.py`. Package managers and forges remain external systems; adapters return evidence rather than becoming alternate planners. No feature requires a pre-existing local fleet, private registry or live service. Unknown identity, private path, unverified version, missing mirror, unsupported syntax, uncertain publication and timeout fail closed with a diagnostic.

Why a spec: no other spec owns release planning and the dependency-floor rollout. Search: `git grep -n "release" -- specs` finds only this spec.

## User Scenarios & Testing

### User Story 1 - Plan a dependency-ordered release (Priority: P1)

The user is a release operator. The outcome is a frozen plan with deterministic stages, a digest and a reason for each edge.

**Acceptance scenarios**

1. **Given** a portable manifest and complete source observations, **When** the operator plans a release, **Then** the plan has deterministic stages and one digest that does not change when the input order changes.
2. **Given** an omitted repository or source, **When** the operator plans a release, **Then** the receipt is incomplete and no executable plan exists.
3. **Given** a two-node cycle or an upward phase edge, **When** the operator plans a release, **Then** a typed diagnostic names the members and no graph is returned.

Independent test: the bound tests of `RM-RELEASE-R001` to `RM-RELEASE-R004`.

### User Story 2 - Execute and recover safely (Priority: P1)

The user is a recovery operator. The outcome is that no stage mutates anything without trusted evidence, and a failed run shows what completed.

**Acceptance scenarios**

1. **Given** a missing artifact, a failed check or a stale predecessor, **When** a dependent stage starts, **Then** it is blocked before any mutation.
2. **Given** a dependency version with only an sdist, **When** readiness is checked, **Then** it fails unless the consumer allow-lists the package.
3. **Given** an injected push failure, **When** the run stops, **Then** a reconcilable partial record exists and success is never reported.

Independent test: the bound tests of `RM-RELEASE-R005` to `RM-RELEASE-R007`.

### User Story 3 - Roll a dependency floor across the fleet (Priority: P2)

The user is a maintainer. The outcome is one auto-merging pull request per repository and a tag after merge.

**Acceptance scenarios**

1. **Given** a dependency and a floor, **When** the maintainer runs `rollout`, **Then** each selected repository gets one pull request with the floor, a verified relock, a patch bump, a changelog entry and regenerated spec status.
2. **Given** a re-run, **When** a rollout pull request is open or merged, **Then** it is adopted and never duplicated.
3. **Given** a merged pull request whose main meets the floor, **When** `release` runs, **Then** it tags once and skips an existing tag.

Independent test: the bound tests of `RM-RELEASE-R009`.

## Requirements

- **FR-001**: Planning derives order from declared and observed edges and rejects cycles (`RM-RELEASE-R001`, `RM-RELEASE-R003`).
- **FR-002**: The census is complete or explicit about what is missing (`RM-RELEASE-R002`).
- **FR-003**: The plan digest is content-addressed and order-independent (`RM-RELEASE-R004`).
- **FR-004**: Stages execute in order with trusted gating and receipts (`RM-RELEASE-R005`, `RM-RELEASE-R007`).
- **FR-005**: All routes share one admission contract (`RM-RELEASE-R006`).
- **FR-006**: Package publication happens only from version tags (`RM-RELEASE-R008`).
- **FR-007**: The dependency-floor rollout is deterministic and resumable (`RM-RELEASE-R009`).

## Success Criteria

- **SC-001**: Permuted equivalent input yields one plan digest (`RM-RELEASE-R004`).
- **SC-002**: A blocked predecessor leaves every dependent repository unmodified (`RM-RELEASE-R005`).
- **SC-003**: Re-running a rollout opens no second pull request (`RM-RELEASE-R009`).

## Edge Cases

- A same-basename package in two repositories: identity is explicit and an ambiguous owner refuses.
- A cargo plain version: read as a caret floor; a node tag or compound range fails closed.
- A lock that resolves below the floor: the row fails and a re-run retries it.
- The forge is unreachable during readiness: the check degrades to the wheel rule.

## Assumptions

- The manifest is a portable input; on-disk Git repositories and hosted artifacts are execution evidence.
- Fleet activation stays closed until canonical metadata and live consumers can cut over together.
- Offline fixtures cover all logic; a live rehearsal on a contributor's own repositories is optional.
