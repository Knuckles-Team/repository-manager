---
name: repository-manager-lane-lifecycle
skill_type: skill
description: >-
  Run one unit of work — one agent session or one person — as an isolated *lane*
  in a repository many other lanes are editing at the same time. Covers opening a
  lane (worktree + partitioned build/test/hook state), staying isolated while
  working, diagnosing a lane that is behaving impossibly, and closing it out.
  Use whenever you are about to edit a repository that other agents or humans are
  also editing, when a test fails in a way that cannot be true, when a build will
  not go green, or when work has gone missing. Do NOT use for landing a branch
  and resolving conflicts (repository-manager-merge-and-reconcile) or for
  workspace-wide sweeps across many repos
  (repository-manager-fleet-scale-operations).
license: MIT
tags: [repository-manager, git, worktree, concurrency, isolation, lane]
metadata:
  author: Genius
  version: '1.0.0'
---
# Repository Manager — Lane Lifecycle

A **lane** is one worker — one agent session or one person — working one
repository in one worktree on one branch. Dozens of lanes run concurrently here.

Every collision this workspace has actually suffered has one shape:

> **A background or global actor mutates state a lane assumed it owned.**

This skill is the working pattern that keeps lanes from destroying each other's
work. It is not advice: each rule below has a check behind it in
`repository_manager/lane_doctor.py`, and every check exists because a *specific*
incident destroyed real work while the rule that would have prevented it was
written down, in prose, in front of the actor who broke it.

## When to use
- You are about to edit a repository other lanes are also editing. **Start here.**
- A test fails in a way that cannot be true; a build will not go green; a merge
  keeps being refused; work you wrote has vanished. **Run `doctor` first**, before
  debugging the code — it answers in under a second.
- You are finishing a piece of work and want it landed.

## When NOT to use
- Landing, gating, conflict resolution, reconciling a moved base →
  `repository-manager-merge-and-reconcile`.
- Sweeping many repositories at once, bulk worktree creation, workspace audits →
  `repository-manager-fleet-scale-operations`.
- Raw worktree verbs with no lifecycle around them →
  `repository-manager-worktree-orchestration`.

## The four arbitration classes

Every shared resource belongs to exactly one class, and each class has exactly one
mechanism. The classification is data, not code, in
`agent_utilities/governance/lane_resources.yaml` — adding a resource is a row,
never a new mechanism. Read it with `agent-utilities lane classify`.

| Class | Rule | Mechanism |
|---|---|---|
| **PARTITION** | Stop sharing it — take your own instance. | `repository-manager --lane env` / `agent-utilities lane env` |
| **APPEND-ONLY** | Stop rewriting it — append to your own fragment; a reconciler folds fragments into one generated view. | `FragmentStore` / `scripts/deferred_registry.py open` |
| **LEASE** | Announce, then **defer**. Exit **75** means another lane holds it. | `agent-utilities lane lease` |
| **READ-ONLY** | Do not mutate it at all. | `agent-utilities lane guard` |

## 1. Open the lane

```bash
repository-manager --lane start --lane-repo <repo> --lane-branch lane/<name> --lane-base main
```

One call does what was three commands of prose, and — the actual point — it
**proves** the isolation instead of asserting it. It returns the worktree path,
the `export` block, and a full preflight report. Creating a worktree is one line
of git; what goes wrong is a lane that *believes* it is isolated while sharing a
cargo target dir, a pre-commit store, and a pytest basetemp with thirteen
siblings.

Then adopt the environment in your shell:

```bash
cd <the worktree it printed>
eval "$(repository-manager --lane env --lane-path . --lane-shell)"
```

That gives you a private `CARGO_TARGET_DIR`, `PYTEST_ADDOPTS --basetemp`,
`TMPDIR`, and `PRE_COMMIT_HOME`.

MCP equivalent: `rm_lane(action="start", repo=…, branch=…, base="main")`.

## 2. Stay isolated while you work

Six rules, each with its replacement command. A prohibition only holds when it
names what to do instead.

### Never edit a canonical checkout
`agent-packages/<repo>` is READ-ONLY for lanes. A background `git reset` on a
canonical tree once discarded ~20 minutes of a lane's work. The `lane-guard`
pre-commit hook **refuses** a non-merge commit authored there; the only carve-outs
are structural (a merge/rebase/cherry-pick in progress, and a pure version bump
whose every staged file is declared in `.bumpversion.cfg`).

→ Work in the worktree `--lane start` gave you.

### Never `git stash`
`refs/stash` is **ONE ref shared by every worktree** of the repository — ~54 of
them here. A `git stash pop` in any lane can consume another lane's entry. Six
collisions, plus four reflexive violations in a single day by actors who had the
rule in front of them. Two different needs hide behind the reflex:

| What you actually want | Do this |
|---|---|
| Read the pristine file while yours is dirty (the common case) | `git show HEAD:<path>` — mutates nothing, works while dirty |
| Actually park work | `git commit -m "wip: …"` on your own branch, or `agent-utilities lane park` (uses `git stash create`, which writes **no** ref, plus your per-lane `refs/lane/<you>/stash`) |

### Never export a shared `CARGO_TARGET_DIR`
A shared cargo target dir does not merely *serialize* concurrent worktree builds
— it **corrupts** them.

```bash
cargo build --target-dir ./target-isolated     # and prune it when you are done
```

`agent-utilities lane bind-cargo` writes `.cargo/config.toml` so the partition
**binds** structurally and needs no export at all.

### Always set your own `PRE_COMMIT_HOME`
This is the newest PARTITION-class resource and the one whose absence is
*silent*. pre-commit's `staged_files_only()` writes your **unstaged** work to a
patch file in the store, `git checkout`s it away so hooks see only staged
content, then restores it in a `finally:`. A crash inside that window loses the
work to an orphaned patch nobody replays (that is D-OB-12's root cause; 27
orphaned patches were found in the shared store). The same directory also holds
pre-commit's SQLite `db.db`, which produces `OperationalError: database is
locked` under concurrent lanes.

`--lane env` sets it. If work has gone missing, look for the patch and replay it:

```bash
repository-manager --lane doctor --lane-path .   # reports shared_store_patches
git apply <the patch path it printed>
```

A patch file alone is **not** proof of a crash — pre-commit never deletes one,
even on success. It is a path to try if work is missing, never a verdict.

### Never `pre-commit --all-files` without the lease
It is LEASE-class: it can destroy unstaged work, and concurrent runs contend for
the same caches.

```bash
agent-utilities lane lease --resource precommit-all-files --operation gate -- \
  pre-commit run --all-files
```

Exit **75** means another lane holds it — defer, do not retry in a loop.

### Commit early and often
Commits are the only thing a working-tree reset cannot take. Necessary, not
sufficient: the unrecoverable window (mid-pre-commit) is the one you cannot
commit from, which is why the rules above exist. **Never `--no-verify`.**

## 3. When something behaves impossibly, run the doctor

```bash
repository-manager --lane doctor --lane-path .
```

Mutates nothing. Returns every check with a verdict, the evidence, and a literal
remedy command. `fail` blocks `finish`; `warn` names a condition that is
legitimate in some lanes and fatal in others, so the decision stays with you.

| Check | The incident behind it |
|---|---|
| `not-canonical` | a background `git reset` on a canonical tree destroyed ~20 min of work |
| `no-worktree-venv` | a worktree-local `.venv` produced **~167 phantom failures** read as real regressions |
| `cargo-partition` | a shared `CARGO_TARGET_DIR` corrupts, not just serializes |
| `precommit-home` | the shared store swallows unstaged work (D-OB-12) |
| `pytest-basetemp` | ~28 concurrent pytest runs on one basetemp made a baseline materially worse — nearly a false regression call |
| `shared-stash-ref` | `refs/stash` is one ref for every worktree |
| `test-runner` | `uv run pytest` silently runs the **SYSTEM** pytest — see below |
| `canonical-clean` | a dirty canonical tree blocks **every** lane's landing, not just yours |
| `merge-queue-config` | a repo declaring no gates is *refused* by the queue, not defaulted |
| `base-drift` | the branch **tip** is not the tree that lands |
| `committed-work` | only commits survive a reset |

### ★ `uv run pytest` is poisoned
In any repository shipping `scripts/uv_workspace.py`, `uv run pytest` silently
resolves the **system** interpreter and its stale packages. It produced ~80
phantom failures that cited the project's own guards; six lanes were burned
before it was found.

```bash
python3 scripts/uv_workspace.py run --all-extras -- pytest <args>
```

Always print `sys.executable` and the package count in the same run: **≈726
packages is the correct environment, ≈44 is the stale one.** A test verdict from
an unproven interpreter is not evidence.

## 4. Close the lane

```bash
repository-manager --lane finish --lane-path . --lane-base main
```

This preflights (blocking), then hands the branch to the serialized merge queue.
Refusing to enqueue a lane that fails its own preflight is not pedantry: the
queue gates a candidate against a freshly computed baseline, the most expensive
thing in the system, and spending that to rediscover an unset `PRE_COMMIT_HOME`
is the wrong order of operations.

**`enqueue` is not a to-do item you must come back to.** A scheduler drains the
queue every ~5 minutes; it lands your branch and prunes both the worktree and the
branch. Watch it, do not babysit it:

```bash
repository-manager --merge-queue status --repo-path .
```

If it is rejected, the rejection names its evidence — take it to
`repository-manager-merge-and-reconcile`.

## Do not do this
- **Never `git branch -D`.** Only `-d`. Its refusal *is* the safety mechanism —
  it is telling you the work is not actually contained in the base.
- **Never merge into the shared base by hand** because the queue is slow. That is
  how two lanes' resolutions orphan each other.
- **Never `--no-verify`**, and never mask a gate to make it green (`noqa`,
  `type: ignore`, `nosec`, `skip`, `xfail` added to a diff is a red flag a
  reviewer will grep for).
- **Never hand-edit a generated view** (`docs/concept_reservations.yaml`,
  `reports/PROGRAM.md`). Write your fragment and regenerate; `lane-guard` refuses
  a hand-edited ledger view.

## Related
- Landing, gates, and conflicts → `repository-manager-merge-and-reconcile`
- Workspace-wide worktree sweeps → `repository-manager-fleet-scale-operations`
- Raw worktree verbs → `repository-manager-worktree-orchestration`
- Mechanism: `repository_manager/lane_doctor.py`,
  `agent_utilities/governance/lanes.py`, `lane_resources.yaml`
