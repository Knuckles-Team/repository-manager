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
- `repositories` — comma-separated project names/paths to scope `rm_projects(action="validate")`;
  omit to target all. **Not `projects`** — that is `rm_workspace(action="maintain")`'s scoping
  field (see below); the two condensed tools use different parameter names for the same idea.
- `validate` returns a **job id**; poll it with `validate_status` (pass `job_id`, `summary`).
- `failed_only` — for `validate`: re-validate only repositories whose most-recent run failed
  (ignored if `repositories` is set; forces re-validation of that set).
- `commit_code` / `commit_message` — for `validate`: after validation passes, stage + pre-commit +
  commit across the targeted repos before any bump.
- `auto_bump` / `auto_push` / `bump_part` — for `validate`: chain a phased version bump and/or push
  once validation passes (see `repository-manager-workspace-release` for the full DAG/consent story).
- `part` — version part to bump for `rm_workspace(action="maintain")` (`major`|`minor`|`patch`).
- `projects` — comma-separated repo names to scope `rm_workspace(action="maintain")` (this tool's
  field IS named `projects`, unlike `rm_projects`'s `repositories`).
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
rm_projects(action="validate", repositories="agent-utilities,gitlab-api")
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
- ★ **`rm_projects(action="validate")` scopes on `repositories`; `rm_workspace(action="maintain")`
  scopes on `projects`.** The two condensed tools do not share a parameter name for the same idea
  — do not assume the field you used to scope one also scopes the other.
- Validation covers pre-commit hooks **and** pytest; a project is only clean when both pass.
- `maintain` bumps are phased/dependency-ordered — `auto_start` begins at the lowest phase with
  pending work; use `dry_run` first on a large workspace.
- `save` overwrites `workspace.yml` — pass an explicit `yml_path` when you don't want the default.
- Any tool's live action set is self-discoverable: `rm_projects(action="list_actions")` /
  `rm_workspace(action="list_actions")` return the current set instead of running anything.

## Related
- `repository-manager-worktree-orchestration` to audit which projects have unmerged/unpushed work.
- `repository-manager-workspace-release` for the DAG-ordered version/floor plan and the consented
  auto-bump/auto-push chain this skill's `validate` action can trigger.
- `repository-manager-development-lifecycle` — the governed entrypoint; per-repo validation is one
  step of that larger lifecycle, not a replacement for it.
- The universal-skills `workspace-validator` workflow composes these `validate` calls to fix all
  project errors until zero remain.
