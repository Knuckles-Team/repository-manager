---
name: repository-manager-workspace-validation
skill_type: skill
description: >-
  Install, build, and validate the managed projects of a workspace via the
  repository-manager MCP server — run pre-commit + pytest validation per project
  and install/build ecosystems. Use when the agent must ensure all projects are
  valid, fix project errors, or run the validation suite. To raise a dependency
  floor across the fleet use `rm_rollout` / `repository-manager rollout`
  (repository-manager-workspace-release). Do NOT use for isolated session worktrees
  (repository-manager-worktree-orchestration) or bulk clone/pull/push
  (repository-manager-bulk-git-operations).
license: MIT
tags: [repository-manager, validation, pre-commit, bumpversion, mcp]
metadata:
  author: Knuckles-Team
  version: '0.1.0'
---
# Repository Manager — Workspace Validation

Install, build and validate the managed **projects** of a workspace over the
repository-manager MCP server. Validation runs each project's pre-commit hooks and pytest
suite and returns a structured pass/fail report. Version bumps and dependency-floor raises
are not done here: they land through pull requests via `rm_rollout` (see
`repository-manager-workspace-release`).

## When to use
- Validate one or all projects (pre-commit + pytest) and get a pass/fail report (`rm_projects_validate`).
- Install or build a project ecosystem (`rm_projects_install` / `rm_projects_build`).
- List workspace projects / their branches (`rm_workspace_list` / `rm_workspace_list_branches`).
- Scaffold, template, or save a `workspace.yml` (`rm_workspace_setup` / `rm_workspace_template` / `rm_workspace_save`).

## When NOT to use
- Isolated per-session branch worktrees → `repository-manager-worktree-orchestration`.
- Bulk clone / pull / push / commit → `repository-manager-bulk-git-operations`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`repository-manager`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `REPOSITORY_MANAGER_WORKSPACE` / `WORKSPACE_PATH` | ✅ | Workspace root |
| `WORKSPACE_YML` | optional | Manifest filename (default `workspace.yml`) |
| `WORKSPACE_REPORTS` | optional | Where validation reports are written |
| `RM_JOB_STALE_SECONDS` | optional | Background-job reap threshold |
| `WORKSPACE_MANAGEMENTTOOL` / `PROJECT_MANAGEMENTTOOL` | optional | Gate the tool tags on |

## Tools & actions
| Tool group | Per-operation tools |
|----------------|---------|
| `rm_projects_*` | `rm_projects_install`, `rm_projects_build`, `rm_projects_validate`, `rm_projects_validate_status` |
| `rm_workspace_*` | `rm_workspace_fleet_evidence_check`, `rm_workspace_list`, `rm_workspace_list_branches`, `rm_workspace_setup`, `rm_workspace_template`, `rm_workspace_save` |

### Key parameters
- `repositories` — comma-separated project names/paths to scope `rm_projects_validate()`;
  omit to target all.
- `validate` returns a **job id**; poll it with `validate_status` (pass `job_id`, `summary`).
- `failed_only` — for `validate`: re-validate only repositories whose most-recent run failed
  (ignored if `repositories` is set; forces re-validation of that set).
- `commit_code` / `commit_message` — for `validate`: after validation passes, stage + pre-commit +
  commit across the targeted repos.
- `yml_path` / `config_dict` / `use_default` — for `setup` / `template` / `save`.

## Recipes
Validate every project (returns a job id), then poll:
```
rm_projects_validate()
rm_projects_validate_status(job_id="<id>", summary=true)
```
Validate only two projects:
```
rm_projects_validate(repositories="scion-sdk,gitlab-api")
```
Preview a dependency-floor rollout across the agents group (no mutation):
```
rm_rollout(action="plan", dependency="scion-sdk", floor="0.6.4")
```
List all workspace projects:
```
rm_workspace_list()
```

## Gotchas
- `validate` runs as a **background job** — use `validate_status` with the returned `job_id`;
  don't expect inline results.
- Validation covers pre-commit hooks **and** pytest; a project is only clean when both pass.
- `save` overwrites `workspace.yml` — pass an explicit `yml_path` when you don't want the default.
- Any tool's live action set is self-discoverable: the served `rm_projects_*` tool list /
  the served `rm_workspace_*` tool list return the current set instead of running anything.

## Related
- `repository-manager-worktree-orchestration` to audit which projects have unmerged/unpushed work.
- `repository-manager-workspace-release` for the dependency-floor rollout (`rm_rollout`:
  plan → apply → status → release), which lands every change through a pull request.
- `repository-manager-development-lifecycle` — the governed entrypoint; per-repo validation is one
  step of that larger lifecycle, not a replacement for it.
- The universal-skills `workspace-validator` workflow composes these `validate` calls to fix all
  project errors until zero remain.
