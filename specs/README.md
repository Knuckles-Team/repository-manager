# repository-manager specifications

These specs follow the ecosystem [spec standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/spec-standard.md): one owning spec per capability, extend an existing spec before creating one, audits and reviews land in the owning spec, one ID and title form, one file set. The `spec-standard` gate enforces it.

This tracked `specs/` directory is the public build contract for **RM** owned work. Every
spec must contain the behavior, architecture, interfaces, tests, quality gates, and acceptance
criteria needed to implement it from this repository. External program notes may inform a draft,
but no private document or local workspace path is required to build or verify a public spec.
Delivery status is computed from the requirement rows, the `Spec:` trailers in Git history and the test bindings.

## Structure

Create `specs/<stable-id>/` with `spec.md` (user outcome, user stories, requirements, acceptance), `plan.md`
(architecture, reuse, interfaces, live wiring, decisions), `test-spec.md` (positive, negative,
integration, quality and release proof), `tasks.md` (ordered implementation and verification),
`requirements.md` (the register defining every requirement ID this spec owns, one row per ID) and
`checklists/requirements.md`. A cross-repository seam adds one `contracts/<seam>.md`, a served row adds
`quickstart.md`, and a row that adds types adds `data-model.md`.
Start from the vended templates in [`.specify/templates/`](../.specify/templates/); pipelines owns them and the
`spec-templates` hook keeps the copy byte-identical. Delivery status is computed, never written:
`pipelines-hook spec-status --write` derives it from `requirements.md`, `Spec:` trailers in Git history and
test bindings into `specs/<dir>/status.json`, which is gitignored. A planned or tested item is not a landed item.
This follows a specify, plan and tasks flow with an explicit test contract. The tracked [constitution](../.specify/memory/constitution.md) records this repository's governing principles.

Each spec names its stable ID and sole owner. Cross-repository work links the other owners'
public specs by stable ID and GitHub URL; each repository documents the complete contracts and
acceptance it owns. Designs inventory existing code and reuse the legal
owner and live wiring before proposing new components. Include CCCC, jscpd, dupehound, KISS,
language-native and repo release gates where applicable, with exact pass evidence and a
no-duplicate-authority check.

## States

States are `SPECIFIED`, `LANDED`, `VERIFIED` and `RETIRED`. The generator sets them. See the [spec standard](https://github.com/Knuckles-Team/pipelines/blob/main/reference/spec-standard.md#states).

## Owner map

| Prefix | Repository specs | Responsibility |
|---|---|---|
| `MEDULLA` | [medulla](https://github.com/Knuckles-Team/medulla/tree/main/specs) | facts, records, storage, query, retrieval, ingest, reasoning, decisions and choices |
| `SCION` | [scion-sdk](https://github.com/Knuckles-Team/scion-sdk/tree/main/specs) | connector contracts, transports, source client, repository transport |
| `MERISTEM` | [meristem](https://github.com/Knuckles-Team/meristem/tree/main/specs) | the in-run harness |
| `ARBORA` | [arbora](https://github.com/Knuckles-Team/arbora/tree/main/specs) | admission, auth, served operations, MCP serving, deployment operations |
| `OVERSTORY` | [overstory](https://github.com/Knuckles-Team/overstory/tree/main/specs) | browser presentation and interaction |
| `PIPE` | [pipelines](https://github.com/Knuckles-Team/pipelines/tree/main/specs) | gates, release rules and the spec standard |
| `RM` | [repository-manager](https://github.com/Knuckles-Team/repository-manager/tree/main/specs) | repository discovery, worktrees, source control and execution |
| `CONN-REPOSITORY-MANAGER` | [repository-manager](https://github.com/Knuckles-Team/repository-manager/tree/main/specs) | this repository's connector content and served tools |

## Contributions

Read this repository's `AGENTS.md` and any contribution guide. Propose `spec.md` first, resolve
architecture and test details in `plan.md` and `test-spec.md`, then implement `tasks.md` in a
dedicated branch or worktree. Link the PR to spec IDs through the `Spec:` commit trailer and run the gates.
Use the [universal-skills spec-generator](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-generator),
[spec-verifier](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-verifier)
and [task-planner](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/task-planner)
skills, together with the [SDD full lifecycle](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development-workflows/sdd-full-lifecycle) workflow.

## Local specifications

- [`RM-MATERIALIZE`](proposal-materialization/spec.md) — governed proposal to Git materialization.
- [`RM-IDENTITY`](fleet-git-identity/spec.md) — repository history and migration receipts.
- [`RM-CONNECTOR`](connector-boundary/spec.md) — connector boundary migration.
- [`RM-GOVERNANCE`](development-governance/spec.md) — repository development governance.
- [`RM-RELEASE`](fleet-release/spec.md) — dependency aware fleet release.
- [`RM-SCM`](source-changesets/spec.md) — git import, export and mirror sync for engine changesets.
- [`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT`](connector-content/spec.md) — the connector serves its packaged content and served tools.
