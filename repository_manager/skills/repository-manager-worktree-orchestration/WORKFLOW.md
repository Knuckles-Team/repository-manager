# Repository Manager Worktree Orchestration

Concurrent multi-session git worktree orchestration via the repository-manager MCP server — create, list, sync, merge, prune, and audit linked worktrees under a shared workspace. Use when the agent must open an isolated branch worktree for a session, list/audit which worktrees are merged / active / stale / dangling, sync a worktree against its base, merge finished work back, or safely prune merged worktrees. Do NOT use for bulk clone/pull/push across repos (repository-manager-bulk-git-operations) or per-project validation/version bumps (repository-manager-workspace-validation).

# Repository Manager — Worktree Orchestration

Manage git **worktrees** for concurrent multi-session development over the
repository-manager MCP server. Each session works a repo in its own worktree on its
own branch under `<WORKTREE_ROOT>/<repo>/<branch>` (shared `.git`, no re-clone),
leaving the canonical checkout on its default branch so a working-tree reset never
disturbs in-flight work.

## When to use
- Open an isolated worktree/branch for a new session (`add`, or `bulk_add` across repos).
- List existing worktrees for one repo or the whole workspace (`list`).
- Classify worktrees as merged / active / stale / dangling before pruning (`audit`).
- Bring a worktree up to date with its base (`sync`), or merge finished work back (`merge`).
- Remove a worktree and optionally its branch (`remove`), or prune stale admin pointers (`prune`).
- Reset a worktree's branch pointer onto another target (`reset_branch`).

## When NOT to use
- Bulk clone / pull / push / commit across repositories → `repository-manager-bulk-git-operations`.
- Install / build / validate / version-bump managed projects → `repository-manager-workspace-validation`.
- Editing files inside a worktree — that is ordinary file work, not this tool.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`repository-manager`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `REPOSITORY_MANAGER_WORKSPACE` / `WORKSPACE_PATH` | ✅ | Workspace root holding the canonical checkouts |
| `REPOSITORY_MANAGER_WORKTREE_ROOT` | ✅ | Root under which linked worktrees live |
| `REPOSITORY_MANAGER_DEFAULT_BRANCH` | optional | Base branch (default `main`) |
| `WORKSPACE_MANAGEMENTTOOL` / `PROJECT_MANAGEMENTTOOL` | optional | Gate the tool tags on |

`MCP_TOOL_MODE` (`condensed`|`verbose`|`both`) selects the condensed `rm_worktree`
tool (used below) vs. the 1:1 verbose actions.

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `rm_worktree` | `add`, `list`, `remove`, `merge`, `sync`, `prune`, `bulk_add`, `audit`, `reset_branch` |

### Key parameters
- `repo` — repo basename (e.g. `agent-utilities`) or absolute path. Omit for `list`/`prune` across all repos.
- `branch` — the worktree branch (each session uses a distinct branch). Required for
  `add`/`remove`/`merge`/`sync`/`reset_branch`.
- `base` — branch to fork from / sync against (default `main`); `into` — merge target for `merge`,
  or the reset target branch for `reset_branch` (default `main`).
- `adopt` — for `add`: move the canonical checkout's uncommitted WIP onto the new branch via a
  **private ref** (never the shared `refs/stash` stack — see the lane-lifecycle skill's stash
  warning; this is the one sanctioned exception to "never `git stash`").
- `stale_days` — for `audit`: unmerged & quiet longer than this is `stale` (default 14).
- `prune_merged` — for `audit`: **DESTRUCTIVE**, removes every `merged` worktree + prunes `dangling` pointers.

## Recipes
Open a session worktree for one repo:
```
rm_worktree(action="add", repo="agent-utilities", branch="feat/my-change", base="main")
```
Audit the whole workspace (read-only), 21-day staleness threshold:
```
rm_worktree(action="audit", stale_days=21)
```
Sync a worktree against main, then merge it back:
```
rm_worktree(action="sync", repo="agent-utilities", branch="feat/my-change", strategy="rebase")
rm_worktree(action="merge", repo="agent-utilities", branch="feat/my-change", into="main")
```
Safely reclaim only merged worktrees:
```
rm_worktree(action="audit", prune_merged=true)
```
Reset a worktree's branch pointer onto another target (e.g. after discarding a bad rebase):
```
rm_worktree(action="reset_branch", repo="agent-utilities", branch="feat/my-change", into="main")
```

## Gotchas
- `audit` classifies: `merged` (clean + captured in base → prunable), `active` (dirty or recent unmerged),
  `stale` (unmerged & quiet), `dangling` (detached/missing pointer). Only `merged`/`dangling` are auto-pruned.
- `prune_merged=true` is destructive — it deletes merged worktrees *and their branches*. Orphaned
  directories (untracked worktree dirs) are report-only and never removed.
- These repos use custom worktree paths; always identify a worktree by its actual `path` from `list`.
- A dirty tree is always classified `active` and `remove` refuses it unless `force=true`.
- Any tool's live action set is self-discoverable: `rm_worktree(action="list_actions")` returns the
  current set instead of running anything — use it if this table and the live server ever disagree.

## Related
- **The lane lifecycle around these verbs** (`rm_lane`: start → doctor → finish) →
  `repository-manager-lane-lifecycle`. Prefer `rm_lane(action="start")` over a bare
  `rm_worktree(action="add")`: it also partitions the build/test/hook state and *proves*
  the isolation, which a bare worktree does not.
- **Landing and reconciling** (`rm_merge_queue`) → `repository-manager-merge-and-reconcile`.
  Prefer the queue over `rm_worktree(action="merge")` into a shared base: the queue gates the
  MERGED tree differentially, lands fast-forward-only under both guards, and prunes for you.
- **Fleet-scale waves and audits** → `repository-manager-fleet-scale-operations`.
- **The governed lifecycle entrypoint** → `repository-manager-development-lifecycle`. These are the
  raw worktree verbs underneath it; prefer the lifecycle skill for one unit of work.
- Enumerating + KG-ingesting repositories (`repository_ingest_repositories`) → the native ingestion tool.
- **Composed by:** the universal-skills `workspace-validator` workflow uses these audits to decide which
  worktrees are safe to prune vs. in-flight.
