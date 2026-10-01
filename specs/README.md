# repository-manager specifications

This tracked `specs/` directory is the public build contract for **RM** owned work. Every
spec must contain the behavior, architecture, interfaces, tests, quality gates, and acceptance
criteria needed to implement it from this repository. External program notes may inform a draft,
but no private document or local workspace path is required to build or verify a public spec.
Delivery status is recorded here with exact merged revision and test evidence.

## Structure

Create `specs/<stable-id>/` with `spec.md` (user outcome, requirements, acceptance), `plan.md`
(architecture, reuse, interfaces, live wiring, decisions), `test-spec.md` (positive, negative,
integration, quality and release proof), `tasks.md` (ordered implementation and verification),
`requirements.md` (the register defining every requirement ID this spec owns, one row per ID),
and `status.json` (machine-readable delivery and acceptance, with a `requirements` array giving
each requirement ID its own `delivery_state` and evidence, alongside the spec-wide
`acceptance_state` and `evidence` array).
Start from [`_template/`](_template/). Keep status and evidence explicit; a planned or tested item
is not a landed item. Put durable evidence links in the spec directory, never local scratch output.
This follows GitHub Spec Kit's specify/plan/tasks flow with an explicit test contract. The tracked [constitution](../.specify/memory/constitution.md) records this repository's governing principles.

Each spec must name its stable ID and sole owner. Cross-repository work links the other owners'
public specs by stable ID and GitHub URL; each repository documents the complete contracts and
acceptance it owns. Designs must inventory existing code and reuse the legal
owner and live wiring before proposing new components. Include CCCC, jscpd, dupehound, KISS,
language-native and repo release gates where applicable, with exact pass evidence and a
no-duplicate-authority check.

## Status and evidence

The tracked `status.json` is the source for the public HTML report. It uses delivery states
`UNKNOWN`, `SPECIFIED`, `BUILDING`, `BUILT`, `LANDED`, `CLOSED`, `DEFERRED`, and `REJECTED`,
and acceptance states `NOT_AUDITED`, `PENDING`, `ACCEPTED`, and `FAILED`. Use
`SPECIFIED/NOT_AUDITED` for a complete design awaiting build; use `UNKNOWN/NOT_AUDITED`
when existing code has not received an exact-revision audit. Prose labels such as `PARTIAL`
or `IN REVIEW` may describe current work, but they do not prove delivery.

`LANDED` requires a public merged-head receipt for the exact owning-repository revision.
`ACCEPTED` additionally requires the checked-in test and consumer or release receipts.
Record public issue, PR, check, and commit links in the owner spec and evidence array.
An obligation can be landed while acceptance remains open. The same rule applies per
requirement: each entry in `status.json`'s `requirements` array counts as delivered only once
its evidence cites a commit merged to the repository's default branch, not a branch tip or a
local build.

## Graph OS owner map

| Prefix | Repository specs | Responsibility |
|---|---|---|
| `EG` | [epistemic-graph](https://github.com/Knuckles-Team/epistemic-graph/tree/main/specs) | Rust database, graph compute, ontology, governed ingestion, durable records and clients |
| `SDK` | [agent-connector-sdk](https://github.com/Knuckles-Team/agent-connector-sdk/tree/main/specs) | connector control, transport, manifests, certification, governed effects |
| `AU` | [agent-utilities](https://github.com/Knuckles-Team/agent-utilities/tree/main/specs) | agent orchestration and control workflows |
| `GRAPHOS` | [graph-os](https://github.com/Knuckles-Team/graph-os/tree/main/specs) | serving composition, fleet, gateway, A2A, deployment operations |
| `WEBUI` | [agent-webui](https://github.com/Knuckles-Team/agent-webui/tree/main/specs) | browser presentation and interaction |
| `RM` | [repository-manager](https://github.com/Knuckles-Team/repository-manager/tree/main/specs) | repository discovery, worktrees, source control and execution |

## Contributions

Read this repository's `AGENTS.md` and any contribution guide. Propose `spec.md` first, resolve
architecture and test details in `plan.md` and `test-spec.md`, then implement `tasks.md` in a
dedicated branch or worktree. Link the PR to spec IDs and update evidence and status only when the
corresponding gates actually pass. Use the [universal-skills spec-generator](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-generator),
[spec-verifier](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-verifier),
and [task-planner](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/task-planner)
and the [graph-os-development](https://github.com/Knuckles-Team/graph-os/blob/main/graph_os/skills/graph-os-development/SKILL.md)
bootstrap skill, together with the [SDD full lifecycle](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development-workflows/sdd-full-lifecycle) workflow.

## Local specifications

- [`RM-MATERIALIZE-001`](proposal-materialization/spec.md) — governed proposal to Git materialization.
- [`RM-IDENTITY-001`](fleet-git-identity/spec.md) — repository history and migration receipts.
- [`RM-CONNECTOR-001`](connector-boundary/spec.md) — connector boundary migration.
- [`RM-GOVERNANCE-001`](development-governance/spec.md) — repository development governance.
- [`RM-RELEASE-001`](fleet-release/spec.md) — dependency aware fleet release.
