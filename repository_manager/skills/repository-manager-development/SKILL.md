---
name: repository-manager-development
skill_type: skill
description: >-
  Deliver a small, source-backed repository-manager change from an isolated
  worktree through focused validation, normal review, and evidence-based
  publication. Use this skill for repository-manager's own MCP adapter, CLI,
  packaging, skills, workflows, and development gates; use the narrower
  lifecycle and release skills for their detailed operations.
license: MIT
tags: [repository-manager, development, mcp, packaging, test, gates]
metadata:
  author: Genius
  version: '1.0.0'
---
# Repository Manager — Development

Repository Manager is the governed workspace and development adapter. Its
Python CLI is implemented in `repository_manager/repository_manager.py`; the
MCP server and its environment defaults live in `repository_manager/mcp_server.py`.
The package exposes the `repository-manager-mcp` console script and the
`agent_connector_sdk.skill_providers` entry point loads skills from
`repository_manager.skills`. Connector contracts and transports belong to
`agent-connector-sdk`; admission, serving, and durable graph state belong to
their owning repositories.

Read `AGENTS.md` and the owning specification before editing. Keep credentials,
provider data, and private workspace paths out of source, logs, skills, and
receipts.

## When to use

- Add or repair a repository-manager skill, MCP surface, CLI behavior, package
  data, workflow, or repository-local development gate.
- Deliver one bounded requirement in a real branch and leave a reviewable
  source, test, and evidence trail.
- Check that a packaged skill remains discoverable by the SDK after a source
  change.

## When not to use

- The complete isolated-lane sequence belongs to
  `repository-manager-development-lifecycle` and
  `repository-manager-lane-lifecycle`.
- Worktree verbs and queue reconciliation belong to
  `repository-manager-worktree-orchestration` and
  `repository-manager-merge-and-reconcile`.
- Workspace-wide validation, version planning, and release sequencing belong
  to `repository-manager-workspace-validation` and
  `repository-manager-workspace-release`.
- Cross-repository fan-out belongs to the fleet-scale and bulk-operation
  skills. Keep this skill's change in this repository.

## Repository source map

- `repository_manager/repository_manager.py` owns the Python CLI and the
  repository-manager implementation used by the console script.
- `repository_manager/mcp_server.py` builds the FastMCP server and exports the
  live action constants. Its registry loads git, project, workspace, gate,
  lane, worktree, merge-queue, build, docs-readiness, concept, and remote
  worker surfaces through the existing registrars.
- `repository_manager/mcp_tools/registry.py` is the deterministic registrar
  seam. Keep a new surface there instead of adding a second discovery path.
- `repository_manager/skills/` is package data (`skills/**` in
  `pyproject.toml`) and is exposed through the
  `repository-manager = "repository_manager.skills"` SDK entry point.
- The server's workspace defaults are source-backed by
  `REPOSITORY_MANAGER_WORKSPACE`, `WORKSPACE_PATH`,
  `REPOSITORY_MANAGER_WORKSPACE_ROOT`, `WORKSPACE_YML`, and
  `REPOSITORY_MANAGER_THREADS`. Tool-family switches are read by the existing
  registrars; do not document a new switch without adding its real source
  contract.

## Rapid delivery: the 5-minute contract

Specs are already designed. Deliver one small, reviewable slice and cite the
owning row in the commit trailer.

1. **Reserve the unit.** Confirm the exact criterion, current `origin/main`,
   changed paths, active worktrees, and open pull requests before editing.
   Start the isolated lane with `rm_lane_start` (or the matching CLI lane
   command) on `origin/main`; do not edit a canonical checkout.
2. **Keep ownership clear.** Preserve the existing MCP contracts, ontology,
   workspace semantics, and runtime behavior. Route shared changes to their
   owning repository rather than duplicating an SDK, pipeline, or registry
   implementation here.
3. **Make the smallest source-backed change.** A skill or documentation
   change must name real source paths and real package-data behavior. A code
   change must reuse the existing registrar, facade, client, and typed action
   contract. Add a focused test for the observable requirement in the same
   slice.
4. **Validate narrowly, then normally.** Use `rm_lane_doctor` before
   submission. Run the focused test with the locked project environment, then
   run the normal pre-commit and pre-push hooks for the changed paths; never
   bypass, suppress, or replace a failing hook. Inspect the complete diff and
   stage only the reviewed allowlist.
5. **Review before landing.** Finish through `rm_lane_finish` and record the
   immutable base, head, tree, changed paths, test result, and hook result.
   Watch `rm_merge_queue_status` while the normal pull request receives
   independent source review and required CI. Merge through the normal route
   only after approval.
6. **Prove publication separately.** A merged source commit is not package,
   Pages, or release evidence. Verify the exact published artifact or live
   Pages source before claiming publication; keep status records generated by
   the canonical pipeline.

For this repository, the focused test should name its requirement with
`pytest.mark.spec`. `PIPE-CONNSPEC-R157.4` is a central pipeline criterion and
does not have a local `specs/` row, so a commit for this exact slice records
`Spec: none` with the central criterion named in the explanatory text. Do not
hand-edit generated status files to promote the row.

## Gates and publication

The checked-in `.pre-commit-config.yaml` owns the repository gate. Run a
focused test first, then the normal hook stages from the locked environment.
The `connector-pr.yml` workflow delegates pull-request gates to the shared
connector pipeline. The Pages workflow calls
`pages_pipeline.yml@main` with `content_source: docs`,
`spec_dashboard_enabled: true`, and `.github/spec-dashboard.json`.
The package workflow is tag-triggered (`v*`) and delegates Python publication
to the shared pipeline with `require-tag: true`, followed by the container
publication. Source, wheel, sdist, and Pages claims each require their own
observed evidence.

If package content requires a release, use the repository's existing
`.bumpversion.cfg` and regenerate the lock file through the canonical release
flow. Review every generated version and lock diff; do not bump an unchanged
package merely to make a skill row look landed.

## Safety rules

- Do not use a shared checkout, `git stash`, `git add -A`, or `git add .`.
- Do not hand-merge branches, invent MCP actions, or route around a refusal
  with a raw shell or an untyped duplicate.
- Do not claim a test, CI run, public wheel, sdist, Pages deployment, or release
  that was not actually observed for the candidate revision.
- Keep receipts and scratch output outside the repository unless the owning
  specification explicitly requires a tracked artifact.

## Related skills

- `repository-manager-development-lifecycle` for the complete plan → lane →
  check → submit → queue-status sequence.
- `repository-manager-gate-execution` for the two-tier gate ledger and narrow
  retests.
- `repository-manager-workspace-release` for dependency-ordered version and
  release planning.
