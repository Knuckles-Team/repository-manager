---
name: repository-manager-bulk-git-operations
skill_type: skill
description: >-
  Bulk, parallel git operations across a whole workspace of repositories via the
  repository-manager MCP server — clone, pull, push, add, commit, pre-commit,
  phased push, run raw git commands, and enumerate every repo across a GitLab
  instance / GitHub org into an ingest manifest (which also natively ingests them
  into the knowledge graph as typed :GitRepository nodes). Use when the agent must
  operate over many repos at once or discover the full repo inventory. Do NOT use
  for single-session worktree branches (repository-manager-worktree-orchestration)
  or per-project install/validate/version bumps
  (repository-manager-workspace-validation).
license: MIT
tags: [repository-manager, git, bulk, enumerate, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Repository Manager — Bulk Git Operations

Fan-out git operations across every repository in a workspace, plus enterprise-scale
remote **enumeration**, over the repository-manager MCP server. Long-running actions
run as background jobs; poll them via the job status path.

## When to use
- Clone / pull / push / add / commit across many repos in parallel.
- Run pre-commit hooks or a gated `commit_code` (hooks then commit) across projects.
- Phased push (push respecting inter-repo dependency phases).
- Run an arbitrary `raw` git command in a repo.
- `enumerate` every repository across a GitLab instance / GitHub org into an ingest
  manifest — and natively ingest them into the KG as `:GitRepository` nodes.

## When NOT to use
- Isolated per-session branch worktrees → `repository-manager-worktree-orchestration`.
- Install / build / validate / version-bump managed projects → `repository-manager-workspace-validation`.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`repository-manager`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `REPOSITORY_MANAGER_WORKSPACE` / `WORKSPACE_PATH` | ✅ | Workspace root holding the checkouts |
| `REPOSITORY_MANAGER_THREADS` / `RM_MAX_WORKERS` | optional | Parallelism for bulk ops |
| `RM_GATE_BEFORE_PUSH` | optional | Run the pre-push validation gate before pushing |
| `GITLAB_URL` / `GITLAB_PRIVATE_TOKEN` | for `enumerate` gitlab | GitLab instance + PAT |
| `GITHUB_TOKEN` / `GH_TOKEN` | for `enumerate` github | GitHub PAT |
| `GIT_OPERATIONSTOOL` | optional | Gate the git-operations tool tag on |

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `rm_git` | `raw`, `clone`, `enumerate`, `pull`, `push`, `phased_push`, `add`, `commit`, `pre_commit`, `commit_code` |

### Key parameters
- `command` — the git command for `raw`; also the VCS selector (`gitlab`|`github`) for `enumerate`.
- `projects` — comma-separated repo URLs (clone) or directory names/paths (pull/push/add/commit);
  for `enumerate`, comma-separated GitLab groups / GitHub orgs (omit for whole instance / your user).
- `message` — commit message for `commit` / `commit_code`.
- `run_precommit` — for `commit_code`: run hooks before committing (default true).
- `phase` / `auto_start` / `target_project` — control `phased_push`.

## Recipes
Pull every repo in the workspace:
```
rm_git(action="pull")
```
Commit code across two repos with pre-commit hooks first:
```
rm_git(action="commit_code", projects="agent-utilities,gitlab-api", message="chore: sweep", run_precommit=true)
```
Enumerate an entire GitLab instance (also KG-ingests as :GitRepository nodes):
```
rm_git(action="enumerate", command="gitlab")
```
Enumerate specific GitHub orgs:
```
rm_git(action="enumerate", command="github", projects="my-org,another-org")
```

## Gotchas
- `clone`/`pull`/`push`/`commit`/`pre_commit`/`commit_code`/`phased_push` return a **job id** and run in
  the background — poll status rather than expecting inline results.
- `enumerate` returns `{count, run_id, manifest, ingested}`; `ingested` is `null` when no KG engine is
  reachable (best-effort, non-fatal).
- `push` honors the pre-push gate when `RM_GATE_BEFORE_PUSH` is set — a failing gate blocks the push.
- `raw` requires `command`; `commit`/`commit_code` require `message`.

## Related
- `repository_ingest_repositories` — the Wire-First native ingestion tool that enumerates a VCS and pushes
  the repos into the knowledge graph as typed `:GitRepository` nodes (the same ingestion `enumerate` triggers).
- `repository-manager-worktree-orchestration` for isolated session branches.
