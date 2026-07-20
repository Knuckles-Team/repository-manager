# Repository Manager Workspace Validation

Install, build, validate, and version-maintain the managed projects of a workspace via the repository-manager MCP server — run pre-commit + pytest validation per project, install/build ecosystems, and drive phased version bumps and maintenance. Use when the agent must ensure all projects are valid, fix project errors, run the validation suite, or bump versions across a dependency- ordered workspace. Do NOT use for isolated session worktrees (repository-manager-worktree-orchestration) or bulk clone/pull/push (repository-manager-bulk-git-operations).

# Repository Manager — Workspace Validation & Maintenance

Install, build, validate and version-maintain the managed **projects** of a workspace
over the repository-manager MCP server. Validation runs each project's pre-commit hooks
and pytest suite and returns a structured pass/fail report; maintenance drives phased,
dependency-ordered version bumps.

## When to use
- Validate one or all projects (pre-commit + pytest) and get a pass/fail report (`rm_projects` `validate`).
- Install or build a project ecosystem (`rm_projects` `install` / `build`).
- List workspace projects / their branches (`rm_workspace` `list` / `list_branches`).
- Scaffold, template, or save a `workspace.yml` (`rm_workspace` `setup` / `template` / `save`).
- Run phased version-bump maintenance across the workspace (`rm_workspace` `maintain`).

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
| Condensed tool | Actions |
|----------------|---------|
| `rm_projects` | `install`, `build`, `validate`, `validate_status` |
| `rm_workspace` | `list`, `list_branches`, `setup`, `template`, `save`, `maintain`, `maintain_status` |

### Key parameters
- `projects` — comma-separated project names/paths to scope; omit to target all.
- `validate` returns a **job id**; poll it with `validate_status` (pass `job_id`, `summary`).
- `part` — version part to bump for `maintain` (`major`|`minor`|`patch`).
- `phase` / `auto_start` — phased maintenance controls; `dry_run` to preview bumps.
- `yml_path` / `config_dict` / `use_default` — for `setup` / `template` / `save`.

## Recipes
Validate every project (returns a job id), then poll:
```
rm_projects(action="validate")
rm_projects(action="validate_status", job_id="<id>", summary=true)
```
Validate only two projects:
```
rm_projects(action="validate", projects="agent-utilities,gitlab-api")
```
Dry-run a patch bump across the workspace:
```
rm_workspace(action="maintain", part="patch", dry_run=true)
```
List all workspace projects:
```
rm_workspace(action="list")
```

## Gotchas
- `validate` and `maintain` run as **background jobs** — use `validate_status` / `maintain_status`
  with the returned `job_id`; don't expect inline results.
- Validation covers pre-commit hooks **and** pytest; a project is only clean when both pass.
- `maintain` bumps are phased/dependency-ordered — `auto_start` begins at the lowest phase with
  pending work; use `dry_run` first on a large workspace.
- `save` overwrites `workspace.yml` — pass an explicit `yml_path` when you don't want the default.

## Related
- `repository-manager-worktree-orchestration` to audit which projects have unmerged/unpushed work.
- The universal-skills `workspace-validator` workflow composes these `validate` calls to fix all
  project errors until zero remain.
