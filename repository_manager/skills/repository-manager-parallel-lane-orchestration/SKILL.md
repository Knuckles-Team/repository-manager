---
name: repository-manager-parallel-lane-orchestration
skill_type: skill
description: >-
  Run a program as one central orchestrator plus many parallel worker agents: brief
  each lane, land its branch into main locally the moment it arrives, prune the branch
  and worktree immediately, and push only once the program is complete. Covers the lane
  briefing contract, land-as-they-arrive merge discipline, provider-before-consumer
  ordering for cross-repo changes, adjudicating contradictory lane reports, and the
  verification rules that catch a lane (or the orchestrator) being confidently wrong.
  Use when dispatching more than one worker agent at a time, when deciding whether a
  returned lane is safe to land, or when reconciling a workspace back to zero branches
  and zero worktrees. Do NOT use for the mechanics of a single merge or conflict
  (repository-manager-merge-and-reconcile), for opening one lane
  (repository-manager-lane-lifecycle), or for release/publish
  (repository-manager-workspace-release).
license: MIT
tags: [repository-manager, orchestration, subagents, merge-queue, worktrees, verification]
metadata:
  author: Genius
  version: '1.0.0'
---

# Parallel lane orchestration

One orchestrator, many workers. Workers build; the orchestrator decomposes, dispatches,
adjudicates, lands, and prunes. The orchestrator does not do the building — its scarce
resource is judgment, not throughput.

## The loop

1. **Decompose** into lanes partitioned **by file**, not by ticket id. Reserved id blocks
   stop id collisions; they do not stop two lanes implementing the same fix twice in the
   same file. If two lanes would touch one file, sequence them instead.
2. **Dispatch** each lane with the full briefing contract below. Route mechanical and
   well-scoped work to a cheaper model; keep the orchestrator on the expensive one.
3. **Land as they arrive.** The moment a lane reports, merge its branch into `main`
   locally, then delete the branch and remove its worktree. Do not batch. A queue of
   unlanded branches becomes a conflict pile, and worktrees left behind are the single
   largest source of workspace drift.
4. **Push at the end**, once the program is coherent — not per lane.

## Lane briefing contract

Every dispatch states all of these, every time. Omitting one reliably produces a lane that
stalls, edits a shared checkout, or reports unverified work as done.

- **Isolation:** take a real `git worktree add <root>/<repo>/<branch> -b <branch> main`.
  Never edit the canonical checkout — a background sync resets it and discards the work.
- **Never** use a harness worktree-isolation flag against a shared multi-worktree repo. It
  writes `core.bare = true` into the *shared* config and every linked worktree of that repo
  starts failing `git status`/`git commit` invisibly.
- **Do not push.** Commit locally; leave the branch for the orchestrator.
- **Never `--no-verify`.** A scoped `SKIP=<hook-ids>` is allowed only for whole-repo,
  non-differential gates failing on pre-existing state, and only with evidence in the commit
  message that the diff touches none of the implicated paths.
- **Do not park on a background watcher.** Waiting on a notification that never fires is the
  most common way a lane silently burns its whole budget. Re-run the command to check state.
- **Test invocation**, verbatim, including the project-specific runner. A bare test command
  that resolves the system interpreter produces confident false verdicts at scale.
- **Commit identity.** Use the repo's configured identity. Never commit as a name that a
  privacy/identity gate derives its banned-token list from at runtime.
- **Report shape:** separate what was *verified live* from what was *inferred from reading
  code*, and say plainly when something is written but unverified.

## Landing rules

**Verify the tree, not the exit code.** A merge command can succeed and still leave the work
off the base branch.

- `git merge <branch>` merges **into current HEAD**. If a canonical checkout has HEAD parked
  on a feature branch, the merge runs backwards — base into branch — and deleting the branch
  then strands the real work. Assert HEAD is the base branch before merging, and confirm
  afterwards with `git rev-list --count <base>..<parked-ref>` equalling zero.
- **Park before you delete.** `git update-ref refs/lane-park/<branch> <branch>` before any
  branch or worktree removal. Then a wrong call costs a lookup, not the work.
- Ancestry is not content. `--is-ancestor` can say yes while the file content is gone. Test
  the tree: `git cat-file -e <ref>:<path>`.
- **Provider before consumer.** When a change spans repos, merge the half that *defines* the
  symbol before the half that *imports* it. There is usually no shared gate across repos, so
  a consumer merged alone fails only at runtime — and a live-mounted deployment can be one
  restart from a crash loop.
- **Do not land a fail-closed change ahead of its precondition.** If a lane's correct
  behaviour is to 403 until some state exists, and the deployment picks up source live,
  merging it early converts a degraded surface into a hard outage. Hold it until the
  precondition lands.

## Adjudicating lanes

Workers are confident and sometimes wrong. Treat a lane report as evidence, not as fact.

- **When two lanes contradict, go to the source yourself.** Do not average them or pick the
  more senior-sounding one. In practice one has usually read a real code path and drawn a
  local conclusion while missing that the decision was already made upstream of it.
- **Verify the premise, not just the conclusion.** A lane can reason impeccably from a
  premise nobody checked. Before acting on a root cause, capture the pre-state that the
  premise asserts — the cheapest possible read that would falsify it.
- **A gate that reports green may be enforcing nothing.** Prove it fails on a known-bad
  input before trusting it. Test it the way a real commit invokes it, including with the
  environment a git hook injects — inherited `GIT_DIR`/`GIT_INDEX_FILE` can make a gate's
  file discovery return an empty set and emit a confident pass over zero files.
- **Grep the diff for silencing.** Agents under gate pressure reach for suppression
  comments, skips, and expected-failure markers to force green.

## Environment failures that impersonate code failures

Recognise these before diagnosing a diff:

- **Hooks SIGKILLed (`signal 9`), a different hook each attempt.** Check swap. A userspace
  OOM killer triggers on swap pressure even with most of RAM free, and a RAM-backed `/tmp`
  full of build scratch is usually what exhausted it. Sum per-process swap: if it is tiny,
  the swap is holding tmpfs pages, and freeing tmpfs does not drain swap by itself.
- **A whole-repo gate failing on paths your diff never touched.** Almost always pre-existing.
  Confirm by scoping the same gate to your changed files.
- **A gate whose banned-token list is derived at runtime** from something you control, such
  as the last commit's author name.

## Reconciling to zero

At program end, and safe to run at any time:

1. Enumerate every first-party repo's branches and worktrees. Exclude vendored/third-party
   checkouts — their `dev`/`develop`/`master` branches are upstream, not yours.
2. Classify: nothing-to-merge, cleanly-mergeable, conflicting.
3. Merge the clean ones, prune the empty ones, and **park** the conflicting ones as refs.
   Parking preserves content while still reaching zero branches and zero worktrees; report
   the parked list rather than quietly leaving them.
4. Watch for repos whose only branch is a feature branch with no `main`/`master` — renaming
   is correct there; deleting would destroy the repo's entire history.
5. Verify by re-enumerating, and verify the merges by tree reachability.

## See also

`repository-manager-merge-and-reconcile` (single-merge mechanics, differential gating,
conflict decision procedure) · `repository-manager-lane-lifecycle` (opening and isolating one
lane) · `repository-manager-worktree-orchestration` (worktree mechanics) ·
`repository-manager-fleet-scale-operations` (workspace-wide sweeps) ·
`repository-manager-workspace-release` (publish).
