---
name: repository-manager-fleet-scale-operations
skill_type: skill
description: >-
  Run development across MANY repositories and many concurrent lanes at once —
  bulk worktree creation, workspace-wide worktree audits and classification
  (merged / active / stale / dangling / orphaned), safe mass pruning, draining
  several repositories' merge queues in parallel, and sizing concurrency against
  the real binding constraint (disk I/O and swap, not the agent cap). Use when
  launching or reclaiming a wave of lanes, when worktrees have accumulated, or
  when deciding how many heavy jobs may run at once. Do NOT use for a single
  lane's lifecycle (repository-manager-lane-lifecycle) or for landing and
  reconciling one branch (repository-manager-merge-and-reconcile).
license: MIT
tags: [repository-manager, git, worktree, fleet, concurrency, capacity, audit]
metadata:
  author: Genius
  version: '1.0.0'
---
# Repository Manager — Fleet-Scale Operations

Operating ~234 repositories with dozens of concurrent lanes. The failure modes at
this scale are different in kind from a single lane's: you do not lose work to a
merge conflict, you lose it to a machine that ran out of memory during a `git
add`, or to a mass prune that deleted a branch someone was still holding.

## When to use
- Launching a wave of lanes across several repositories.
- Reclaiming accumulated worktrees and branches.
- Draining more than one repository's merge queue.
- Deciding how many heavy jobs may run concurrently.

## When NOT to use
- One lane's lifecycle → `repository-manager-lane-lifecycle`.
- Landing/reconciling one branch → `repository-manager-merge-and-reconcile`.

## ★ The binding constraint is disk I/O and swap — not the agent cap

The instinct is to size a wave by how many agents you may run. That is the wrong
resource. Measured at peak on this host: load ~26 on 24 cores, **swap 100%
exhausted**, 14 concurrent `uv sync` against a 112 GB cache — and a plain
`git add` died with `Bus error`. That is data loss caused by scheduling, not by
any lane doing anything wrong.

**Cap HEAVY lanes at ~3–4 concurrently, independent of total lane count.** Heavy
means anything that does bulk I/O or bulk compilation:

| Heavy (cap 3–4) | Light (costs almost nothing) |
|---|---|
| `uv sync` / any dependency resolution | editing files |
| `cargo build` / `cargo check` | reading, searching, KG queries |
| `pre-commit run --all-files` | `git commit`, `git status` |
| a full `pytest` sweep | `--lane doctor`, `--merge-queue status` |
| container image builds | writing docs/skills |

Light lanes are nearly free; run as many as you have work for. The mistake is
treating all lanes as equivalent and sizing the whole wave by the heaviest one —
or worse, by the agent cap, which measures nothing physical.

Two corollaries:

- **Serialize the LEASE-class heavies.** `uv sync`/`uv lock`,
  `pre-commit --all-files`, and any venv swap are already LEASE-class *because*
  of this. Run them through the lease and let contenders defer:
  `agent-utilities lane lease --resource dependency-lock --operation relock -- uv lock`.
  Exit **75** = another lane holds it.
- **Prune build artifacts at verify time, not at the end.** A shared cargo target
  dir has already hit `ENOSPC` mid-wave. Each lane builds into
  `./target-isolated` and removes it when its gate is green.

## Opening a wave of lanes

```bash
# one branch name across every workspace repo (or a named subset)
rm_worktree(action="bulk_add", branch="lane/<wave>-<name>", repos="agent-utilities,epistemic-graph")
```

Then, per repository, adopt the isolation and prove it:

```bash
repository-manager --lane doctor --lane-path <worktree>
eval "$(repository-manager --lane env --lane-path <worktree> --lane-shell)"
```

Do this per worktree, not once for the wave: `CARGO_TARGET_DIR`, `TMPDIR`,
`PYTEST_ADDOPTS --basetemp` and `PRE_COMMIT_HOME` are all **per-lane** — a wave
that exports one set of values has partitioned nothing.

## Auditing what exists

```bash
rm_worktree(action="audit", stale_days=21)        # read-only, whole workspace
```

The classification, and what each class means operationally:

| Class | Meaning | Safe to reclaim? |
|---|---|---|
| `merged` | clean **and** captured in the base | ✅ yes — this is the only auto-prunable class |
| `active` | dirty, or unmerged and recently touched | ❌ never — someone is in it |
| `stale` | unmerged and quiet longer than `stale_days` | ⚠ **review**, never bulk-delete — unmerged means the work exists nowhere else |
| `dangling` | detached or missing admin pointer | ✅ the pointer is prunable; no work is at risk |
| orphaned dir | an untracked directory under the worktree root | report-only, never removed automatically |

★ **A dirty tree is always `active`.** That is deliberate: uncommitted work is
exactly the work that exists in no other place. `remove` refuses a dirty tree
unless forced, and forcing it is how a wave eats a lane.

★ **`stale` is not `abandoned`.** An unmerged branch quiet for three weeks may be
blocked, deferred, or waiting on a decision. Reclaim `stale` only after reading
what is on it — `git log --oneline <base>..<branch>` — and never in bulk.

## Reclaiming safely

```bash
rm_worktree(action="audit", prune_merged=true)    # DESTRUCTIVE: merged + dangling only
```

What makes this safe is not the flag; it is the guard behind it. The prune
re-checks the merge-base **at delete time** (not at wave start — a branch can
gain commits while the wave runs), writes a `refs/lane-backup/<branch>` anchor
before deleting, uses `git branch -d` and **never** `-D`, and refuses any
worktree holding uncommitted work.

★ **Never `git branch -D` at fleet scale.** `-d`'s refusal is the signal that a
branch is not actually contained in the base. At one-lane scale that is an
inconvenience; across a wave it is the only thing standing between you and
deleting dozens of branches whose work exists nowhere else.

Prefer letting the merge queue prune. A landed candidate's worktree and branch
are removed by the queue itself, under the same guards — so a healthy wave leaves
almost nothing to reclaim by hand. Worktrees accumulating is usually a symptom
that branches are not landing, and the fix is upstream of the prune.

## Draining several repositories' queues

Each repository's queue is independent **by construction**, not by a `repo` key
someone must remember to set: the lease, the candidate store, and the scratch
partitions all resolve through that repository's own `--git-common-dir`. So this
is safe:

```bash
repository-manager --merge-queue run --repo-path /path/to/agent-utilities &
repository-manager --merge-queue run --repo-path /path/to/epistemic-graph &
```

But **a queue drain is a heavy lane** — it runs the repository's full gate set
twice (merged tree + baseline). Count each concurrent drain against the 3–4 heavy
cap. Exit **75** from any of them means another runner already holds that
repository's `reconciliation-merge` lease: defer, do not retry in a loop.

Prefer the scheduler to manual drains. A timer drains every ~5 minutes, which
naturally serializes and keeps the drain off your concurrency budget entirely.

## The dirty-canonical sweep

One dirty canonical checkout blocks **every** lane's landing in that repository,
not just its own — both the canonical guard and the queue's land step refuse a
dirty canonical tree, **including an untracked-only one**. At fleet scale this
shows up as "the queue mysteriously stopped landing anything for repo X."

```bash
repository-manager --lane doctor --lane-path <any worktree of the repo>   # canonical-clean
```

Fix it by committing or moving the files out. ★ Never `git checkout` / `git
clean` a canonical tree to unblock a queue — that is precisely the mutation that
destroyed ~20 minutes of a lane's work. Route every tree-mutating verb through
the guard:

```bash
agent-utilities lane guard --reset <path> --owner <you>
```

It refuses any tree holding uncommitted work you do not own. Skip that tree;
never force. ⚠ A lease binds only actors that take it — an unwrapped external
process still races. That gap is real and open.

## ★ Landing at fleet scale is a live deploy

Fleet pods `hostPath`-mount the canonical tree over `site-packages`, so a wave
that lands ten branches has performed ten deploys. Check runtime compatibility
against the **deployed images**, not just your venv, and expect a config-contract
change to need a migration before the fleet can start.

## Related
- One lane's lifecycle → `repository-manager-lane-lifecycle`
- Landing and conflicts → `repository-manager-merge-and-reconcile`
- Raw worktree verbs → `repository-manager-worktree-orchestration`
- Bulk clone/pull/push across repos → `repository-manager-bulk-git-operations`
