# The universal merge queue (CONCEPT:RM-MERGE-QUEUE)

> One serialized merge queue, for **any** git repository, driven by huge concurrent
> waves of agents and humans. The mechanism lives here; the gates live in each
> repository's own `.mergequeue.yaml`.

## Why it moved

`agent_utilities/governance/merge_queue.py` is mature and battle-tested, but it
could serve exactly one repository: its gates were pytest/ruff/contract-script
specific by construction. The proof that this was mis-homed is empirical —
**epistemic-graph has no merge queue at all**, so its lane had to hand-apply the
discipline from memory: verify the *merged tree* with `cargo check
--all-features`, fast-forward only, never merge in the canonical checkout.

repository-manager already owned the rest of the machinery:
`WorktreeManager` (over a backend-agnostic `GitLike` Protocol), `canonical_guard`
and `prune_guard` — the exact guards the au queue had to **reimplement inline**
when repository-manager turned out not to be importable (D-ORC-21). Only the
queue layer was in the wrong package.

## Mechanism vs gates

```
MECHANISM  (repository_manager/merge_queue.py — generic, this file)
  enqueue / status / withdraw / run / config
  per-repo candidate store        differential gating vs a base ref
  regenerate-on-land              guarded prune
  the reconciliation-merge lease  fold-by-recorded_at ordering
  optimistic batching + bisection fast-forward-only landing

GATES      (<repo>/.mergequeue.yaml — declarative, per project)
  a command, a tier, a timeout, and HOW to compare its result to the base
```

**The queue must not know what a gate IS — only how to run one and compare its
result against the base ref.** A repository that declares no config is *refused*,
not defaulted: "declared no gates" and "has no queue configured" must not be the
same value.

```mermaid
flowchart TD
    L1[lane A worktree] -->|enqueue| S[(candidate store<br/>append-only fragments<br/>in the repo's git-common-dir)]
    L2[lane B worktree] -->|enqueue| S
    L3[lane N worktree] -->|enqueue| S
    S --> R{{run — holds the repo's<br/>reconciliation-merge LEASE}}
    R --> C[merge-tree --write-tree<br/>→ commit-tree<br/>NO working tree touched]
    C --> G[materialize a throwaway<br/>detached worktree]
    G --> Y[".mergequeue.yaml<br/>read from the MERGED tree"]
    Y --> F[run each fast-tier gate]
    F --> B[run the SAME gate on the base ref<br/>→ differential comparison]
    B -->|new signal| REJ[reject with the evidence<br/>candidate stays on its branch]
    B -->|only pre-existing| FF[git merge --ff-only<br/>under BOTH guards]
    FF --> P[guarded prune:<br/>worktree + branch -d + anchor]
    REJ -.->|batch >1| BI[bisect: log2 N extra runs]
    BI --> C
```

## The five behaviours carried over, and what each one cost

| # | Behaviour | Why it exists |
|---|---|---|
| 1 | **Differential gating** — reject only failing signals *not* present on the base ref | `main` is legitimately red. An absolute gate deadlocked the queue and stranded **19 branches**; a branch that fixed 21 of 30 failing tests was rejected because 9 remained. **A baseline that cannot be produced REFUSES the candidate — never allow-all.** |
| 2 | **Fold by `recorded_at`** | Resolving cross-lane duplicates by fragment order prefers whichever *lane name* sorts last. A candidate enqueued by `lane-foo` and landed by `canonical` folded to the stale record and reported `queued` forever (D-F6-1/D-CVG-9). |
| 3 | **Regenerate-on-land** | With ~76 candidates on one base, nearly every one conflicts on a purely-derived file where there is no real disagreement. Regenerate from the *already-merged* tree; `--theirs` silently drops a side. |
| 4 | **Guarded prune** | Merge-base re-checked *at delete time*, a `refs/lane-backup/<branch>` anchor written first, `git branch -d` **never** `-D`, and any worktree holding uncommitted work refused. |
| 5 | **Honest degradation** | Every refusal names its reason. No path reports success it did not verify. |

## `.mergequeue.yaml`

```yaml
base: main
batch_size: 8
environment_signature: ["cargo", "--version"]   # busts the baseline cache on a toolchain change

gates:
  - name: cargo-check
    command: [cargo, check, --all-features, --message-format, short]
    tier: fast              # fast = inside the queue; slow = declared, run post-merge
    timeout: 1800
    baseline_timeout: 3600  # the base run is no smaller and contends differently (D-MW-10)
    compare: lines          # exit | lines | pytest-ids
    keep_lines: ['^error', '^warning']   # only these participate in the diff
    ignore_lines: ['generated [0-9]+ warnings?']
    when_changed: ['**/*.rs', 'Cargo.toml']
    on_timeout: fail        # fail | defer

generated_files: [docs/index.md]
regenerate: [["python3", "scripts/gen.py"]]
```

### `compare` — the three shapes failure actually takes

* **`pytest-ids`** — an **ID-level** compare. A failing id is permitted only when
  that *exact* id already fails on the base. Never by file, module, pattern or
  count: an id-level compare is the only shape that cannot be gamed into masking
  a real regression. A pytest exit outside `{0,1,5}` is *unreadable*, not
  "zero failures", and is refused outright.
* **`lines`** — a **line-level** compare of normalized output. Catches a genuinely
  new diagnostic even when the base was already red for something else. The tree
  path is substituted out before diffing (the merged run and the base run happen
  in two different throwaway worktrees, so every absolute path differs);
  `keep_lines` is what makes this usable for a chatty tool — `cargo`'s
  `Compiling`/`Finished in 3.4s` lines differ on every run and would otherwise
  read as new violations on every candidate.
* **`exit`** — **script granularity**, for a tool that prints one static message
  (or nothing) regardless of *why* it failed. Reported as exactly that much
  precision, never dressed up as more.

### `environment_signature` and the baseline cache

The cache is keyed on `(base_sha, gate name, command, compare, environment)`. If
the environment cannot be fingerprinted it reads `unpinned`, and an unpinned
environment **disables the cache** rather than keying it on a fiction — a stale
baseline computed against a toolchain that no longer exists is worse than a slow
one (D-MQD-1).

## Pre-commit safety (D-ORC-37)

`pre_commit/staged_files_only.py` writes your **unstaged** changes to a patch
file, `git checkout`s them away so hooks see only staged content, then restores
them in a `finally:`. `patch_dir` **is** the pre-commit store directory
(`commands/run.py` → `staged_files_only(store.directory)`), which defaults to a
single shared `~/.cache/pre-commit` for every lane on a host. Two hazards follow:
a crash inside the window loses the work to an orphaned patch nobody replays
(D-OB-12), and the same directory holds pre-commit's SQLite `db.db`, which
produced `OperationalError: database is locked` under concurrent lanes.

Four rules, all enforced rather than documented:

1. **Every lane gets its own `PRE_COMMIT_HOME`** — now a PARTITION-class resource
   in agent-utilities' `lane_resources.yaml`, resolved by `partitioned_paths()`
   and exported by `agent-utilities lane env`. One change fixes both hazards.
   This queue sets it for every gate run too (`run_fast_gates`).
2. **Never run pre-commit against a tree holding someone else's uncommitted
   work.** `refuse_precommit_on_dirty_tree()` is called before any gate whose
   command runs `pre-commit`. A clean tree has no patch-restore window at all.
3. **Never `pre-commit --all-files` on a canonical checkout without the
   `precommit-all-files` LEASE** (already LEASE-class).
4. **Treat a crashed pre-commit as a data-loss incident, not a failed check.**
   `lanes.orphaned_precommit_patches()` classifies each patch file as
   `restored` / `ORPHANED` / `in-progress` / `unknown` — a patch file alone is
   *not* proof of a crash, because pre-commit never deletes one even on success —
   and `lane env` / `lane status` surface any `ORPHANED` one loudly **with its
   path**, so the work can be replayed with `git apply <path>`.

## Safety on the canonical tree

repository-manager has destroyed work on a canonical tree before: a background
`git reset` discarded ~20 minutes of a lane's work, which is why the standing
rule is *never edit a canonical checkout, always use a worktree*. Making it the
merge **driver** raises the stakes, so `land()` holds **both** guards:

* `lanes.guarded_tree_mutation` — refuses a tree holding uncommitted work another
  lane owns;
* `canonical_guard.guarded_canonical_mutation` — takes repository-manager's own
  canonical `flock` lease *and* re-checks `git status --porcelain` under it, so a
  land can never interleave with repository-manager's background sync.

That guard is **proven**, not assumed: a clean tree is allowed, a tracked
modification is refused, an **untracked-only** tree is refused, and a lease held
by another process is refused cross-process — see
`tests/test_canonical_guard.py` and the queue's own
`test_prune_refuses_a_worktree_holding_uncommitted_work` /
`test_prune_removes_a_clean_worktree_and_anchors_the_branch` pair (the positive
half exists so the refusal test cannot be vacuous).

## Surfaces

```bash
# CLI — one repository per invocation, selected by --repo-path
repository-manager --merge-queue enqueue --repo-path /path/to/repo
repository-manager --merge-queue status  --repo-path /path/to/repo
repository-manager --merge-queue run     --repo-path /path/to/repo
repository-manager --merge-queue config  --repo-path /path/to/repo   # validate the gates

# Standalone driver — suitable as a systemd/CronJob ExecStart, no CLI import
python -m repository_manager.merge_queue run --path /path/to/repo
```

MCP: **`rm_merge_queue`** with `action` ∈ `enqueue|status|withdraw|run|config`
and `repo_path` selecting the repository. Both surfaces are thin marshallers over
`merge_queue.dispatch()` — one action core, so they cannot drift (pinned by
`test_the_mcp_tool_is_registered_and_declares_every_action`).

**Exit 75 (`EX_TEMPFAIL`) means another runner holds this repository's
`reconciliation-merge` lease — defer, do not retry in a loop.**

## Cross-project independence

The lease, the candidate store and the scratch/basetemp partitions all resolve
through `lane_scope(path)`, which is anchored on that repository's own
`--git-common-dir`. So two repositories' queues are independent **by
construction**, not by a `repo` key someone must remember to set, and draining
agent-utilities and epistemic-graph concurrently is safe. No new arbitration
class was introduced.

## Migration — the live au queue must not break

The au queue is **actively landing branches**. The migration is therefore
strictly additive and reversible, and each step is independently valuable:

| Step | Action | Risk |
|---|---|---|
| 1 ✅ | Ship the generic queue + presets + CLI + MCP here. `agent_utilities.governance.merge_queue` is **untouched**. | none — no au code path changes |
| 2 ✅ | Land the `PRE_COMMIT_HOME` PARTITION fix in agent-utilities (additive: a new `PartitionedPaths` field, a new `lane_resources.yaml` row, a new `lane env` export). | none — nothing reads the field yet |
| 3 | Adopt for **epistemic-graph first** — copy `mergequeue_presets/epistemic-graph.mergequeue.yaml` to its root and drive it. eg has **no queue today**, so there is nothing to break, and it is the honest first proof at production scale. | low |
| 4 | Copy `mergequeue_presets/agent-utilities.mergequeue.yaml` into agent-utilities and run **both** queues against it in shadow: `--merge-queue config` and a `--queue-no-prune` drain on a scratch clone, comparing verdicts candidate by candidate. | low — read-mostly |
| 5 | Cut over: `agent_utilities.governance.merge_queue` becomes a thin shim delegating to `repository_manager.merge_queue` **when importable**, keeping its current implementation as the fallback. Note the dependency direction — repository-manager depends on agent-utilities, so au importing repository-manager must stay optional (that asymmetry is exactly what D-ORC-21 recorded). | medium — do this only after step 4 agrees |
| 6 | Retire the au implementation once the shim has landed real batches, per *No Legacy* (migrate-and-delete, no deprecation window). | — |

**Do not skip to step 5.** Placing `.mergequeue.yaml` in a repository is what
switches that repository over, so the presets shipped here are inert until
copied — which is what makes steps 1-2 zero-risk.

## Deployment — what the MCP server needs

The long-running `repository-manager-mcp` streamable-http server in k8s is the
intended host (replacing the stopgap systemd timer). Two blockers must clear
first; see `k8s/README` in `services/repository-manager-mcp/` and the register
entries D-ORC-36/D-ORC-38.

1. **The pod is in CrashLoopBackOff** and cannot serve anything. Note the
   reported cause is incomplete: the pod reports `No module named
   'fastmcp.server.extensions'`, but on this host the same module
   (`agent_utilities/mcp/tasks_extension.py`) *also* fails with `cannot import
   name 'MCPError' from 'mcp.shared.exceptions'` (the installed `mcp` spells it
   `McpError`). The defensive-import fix must cover **both** symbols.
2. **The pod cannot reach any repository.** Its only mounts are three *package*
   directories into `site-packages` plus the CA bundle — there is no
   `/home/apps/workspace` mount, so the server cannot see a single git tree. The
   queue needs **read/write** access to the canonical checkouts (it moves refs
   and removes worktrees), so a read-only mount is not sufficient:

   ```yaml
   volumeMounts:
     - mountPath: /home/apps/workspace
       name: workspace
       # NOT readOnly: the queue fast-forwards refs, writes the arbitration dir,
       # and removes landed worktrees.
   volumes:
     - name: workspace
       hostPath:
         path: /home/apps/workspace
         type: Directory
   ```

3. **Placement is a deliberate decision, not a default.** The deployment already
   uses `hostPath` with **no `nodeSelector`** — if it ever reschedules, those
   paths vanish; it works today only because it happens to sit on the right node.
   Adding a workspace mount makes that latent bug load-bearing. The trade must be
   named rather than resolved unilaterally: a `nodeSelector`+`hostPath` workload
   that *cannot* reschedule is exactly what took cluster DNS down when its node
   died. Options, in the order they should be considered:
   - **(a)** Pin with `nodeSelector` to the workspace host and accept that the
     queue is unavailable while that node is down (the queue is not on any
     serving path, so this is survivable — unlike DNS).
   - **(b)** Serve the trees over the existing NFS export instead of `hostPath`,
     so the pod can reschedule. Costs NFS semantics on git operations.
   - **(c)** Leave the MCP surface read-only (`status`/`config`) and keep `run`
     on a host-local unit next to the trees.

   Recommendation: **(a) plus an explicit `PodDisruptionBudget`-free,
   single-replica declaration**, because the queue already serializes on a
   host-local `flock` lease and cannot be safely replicated anyway. Decide
   before mounting, not after.
