# Getting build/CI hosts off shared-NFS git repos

## Incident this responds to (2026-08-13)

R820 (a live k8s worker node, 10.0.0.13) had a kernel thread
(`[10.0.0.12-manager]`, the NFSv4 client's per-server state-manager kthread)
pinned at ~98% CPU for 3.5+ hours, spinning in
`nfs_server_reap_expired_delegations` / `nfs_client_for_each_server`. Root
cause, confirmed against live server state
(`/proc/fs/nfsd/clients/<id>/states` on RW710, the NFS server / dev host,
10.0.0.12):

- R820's NFSv4 client held **555,965 outstanding read delegations** against
  RW710 — one per file it had opened and not yet had the delegation
  returned for. R710 (a comparably-loaded host) held 11,582; R510 held
  158,047. Delegation filenames sampled from the server's client-state dump
  are exactly what heavy build/test activity produces: `.fingerprint`
  (Cargo), `site-packages`/`node_modules` (Python/JS deps), `objects` (git
  object stores), `popen-gw*` (pytest-xdist worker temp dirs) — spread
  across dozens of repos.
- The delegation count was **static** (unchanging over a 20s sample) while
  the reaper thread burned CPU continuously — a livelock, not active churn:
  the state-manager thread was spinning over its own delegation list without
  making net progress, most likely because the list had grown past whatever
  scale the client's O(n)-per-scan reap logic tolerates.
- This wedged more than the mount it was scanning: plain `ls`/`df`/`stat`/
  `du`/`lsof`/`mount.nfs` invoked anywhere near the affected server sat in
  `D` state, because the reaper appears to serialize other RPCs to that
  server while it runs. That is the mechanism behind this workspace's other
  already-logged NFS symptoms: a `git merge` that timed out mid-checkout
  leaving files written but HEAD unmoved, a `git commit` that took 10+
  minutes, `index.lock`/partial-index incidents, and epistemic-graph test
  "deadlocks" root-caused across ~70 attempts to host I/O saturation
  (`fdatasync` stuck in `D` state) rather than an actual lock cycle.
- Remediation taken live against R820 (see the session report for the full
  transcript): confirmed via `fuser`/CWD scan that nothing was actively
  using `/home/apps/workspace` or `/home/apps/worktrees` (the two
  NFS-mounted, dev-only mounts on R820 — distinct from the k8s node's own
  `kubelet` NFS volume mounts for the live `agent-webui` pod, which use
  independent local mountpoints and were **not** touched), then `umount -l`
  (lazy) both and stopped their `systemd` `.automount` units to stop them
  silently re-mounting. This is fully reversible
  (`systemctl start home-apps-workspace.automount home-apps-worktrees.automount`).
  It did **not** stop the CPU spin, because the wedged NFSv4 **client**
  (one state-manager thread per server, shared across every mount to that
  server) also serves the pod's own read-only NFS volumes
  (`webui-deps`, `workspace`, `au-src`, `universal-skills-src`,
  `agent-webui-src`) — the only way to fully clear the spin is to force
  that client's state to expire (`echo expire >
  /proc/fs/nfsd/clients/<id>/ctl` on the server, an admin interface kernel
  NFSD exposes for exactly this), which resets state the live pod's mounts
  also depend on. That is a production-affecting action and was
  **deliberately left for the user to authorize**, not taken unilaterally.

## Why repositories don't belong on shared NFS at all

This is bigger than one livelock. Even without it, NFS is a bad substrate
for git specifically:

- Git is dominated by small-file metadata operations — `stat`, `open`,
  rename-based lockfile updates (`index.lock`, ref updates) — each of which
  is a network round trip over NFS instead of a local syscall. Latency that
  is invisible locally multiplies directly into wall-clock time for
  `status`/`commit`/`checkout`.
- NFSv4 delegations exist to let a client cache reads/writes without asking
  the server every time — which is exactly why heavy git/build I/O
  (thousands of small files touched per test run) generates the delegation
  volume this incident traced back to. That scaling failure mode is not a
  one-off misconfiguration; it is what this access pattern does to NFSv4
  delegation state in general.
- `index.lock`'s rename-based mutual exclusion is not reliably atomic over
  NFS the way POSIX rename is expected to be locally, which is the likely
  contributor to this workspace's recurring `index.lock`/partial-index
  incidents.
- Sharing one `.git` across hosts means concurrent writers to the same
  `config`, `refs/stash`, and index — independently identified in this
  workspace as the root of the `core.bare` corruption incidents
  (`EnterWorktree` writing `core.bare=true` into a shared
  `$GIT_COMMON_DIR/config`) and stash hazards. A shared `.git` is a shared
  mutable resource with no coordination protocol; per-host clones remove the
  hazard structurally instead of policing it.

## Decision: per-host local clones over SSH, not NFS, not rsync

**Git is itself a distributed sync tool.** The dev host (RW710) stays the
canonical checkout for interactive work; a build/CI host pulls an immutable
commit via `git clone`/`fetch`/`checkout` over SSH onto its own local disk,
runs its work there, and the result is reported back — no shared
filesystem, no shared `.git`, no NFS mount for the repository itself.

**Where this already exists in `repository-manager`.** RMDD-15
(`repository_manager.remote_execution`) already built exactly this, and
P0.7's `dispatch_build` (`repository_manager.remote_worker_actions`) already
wires it end to end:

1. `register_worker` declares a host's weighted capacity plus, critically,
   its **authorized, non-shared, per-repository local worktree root** —
   never a shared NFS path.
2. `dispatch_build` takes a `repository_id` + an **immutable 40-hex commit
   SHA** (never a branch — a moving ref would make "what actually built"
   ambiguous) and, over `TunnelSSHExecutor` (tunnel-manager's real SSH
   primitive, not a hand-rolled `ssh` invocation):
   - `git clone --no-checkout` the origin onto the host's authorized local
     root,
   - `git fetch --depth 1 origin <sha>` (shallow — pulls exactly the one
     commit's tree, not full history, so the network/disk cost per
     dispatch stays small even for a large repo),
   - `git checkout --detach <sha>`,
   - independently re-derives cleanliness and HEAD identity afterward
     (`git status --porcelain` / `git rev-parse HEAD` through the same
     executor — it never trusts its own claim that staging succeeded),
   - then runs the caller's fixed command in that freshly materialized
     worktree and returns the outcome.

**This was validated live against R820 during this incident** (the exact
host the pathology was on), entirely bypassing its wedged NFS mount:

```
register_worker(host_id="r820", inventory_alias="R820",
                 repository_roots={"repository-manager": "/home/apps/build-worktrees"})
dispatch_build(host_id="r820", repository_id="repository-manager",
                origin="genius@10.0.0.12:/home/apps/workspace/agent-packages/agents/repository-manager",
                tree_sha=<HEAD>, command=["git","log","-1","--format=verified-remote-build: %H %s"])
=> staged.destination = "/home/apps/build-worktrees/repository-manager-5f49bf5c4081"
=> build.stdout_tail  = "verified-remote-build: 5f49bf5c408117fd3062b362ba17427ef1ddc470 Merge branch 'feat/rm-invariants-and-hosts'"
```

`/home/apps/build-worktrees` is genuinely local (`/dev/sda3`, ext4) — not
one of the NFS mounts. `findmnt --target /home/apps/build-worktrees` on
R820 confirms it resolves to the local `/home` filesystem, not
`10.0.0.12:...`.

**What this fixed slice found and closed.** `dispatch_build` was fully
implemented and unit-tested at the `remote_worker_actions.dispatch()` layer
(`tests/test_p07_dispatch_build.py`, `tests/test_p07_build_host_dispatch.py`
— including a prior live validation against R820, per that test module's
own docstring) but **neither adapter exposed it**: the CLI's
`--remote-workers` `choices=[...]` list and the `rm_remote_workers` MCP
tool's parameter set both predated the action and had never been updated —
so it was reachable only via a direct `remote_worker_actions.dispatch(...)`
Python import, not through the CLI or MCP surface a real build pipeline
would actually call. This lane adds `dispatch_build` to the CLI's
`choices=[...]` and adds the missing `command`/`workdir` parameters to
`rm_remote_workers`, with CLI/MCP parity tests
(`tests/test_rmdd20_remote_worker_surfaces.py`) proving both adapters now
reach it identically, plus a real subprocess-level test proving `argparse`
itself accepts the choice (the earlier register/recheck parity tests all
bypass `argparse` by constructing the CLI's `Namespace` directly, which is
exactly why this gap went unnoticed).

## Honest assessment of rsync

The user explicitly asked about rsync, so: it is a good, cheap tool for
**one-way distribution of read-mostly artifacts** — build caches, wheels,
container layers, vendored dependency snapshots, anything where "make
host B look like host A's copy of this directory" is the whole
requirement and only one side ever writes. It is meaningfully cheaper than
NFS for that (one transfer, then done — no ongoing mount, no live protocol
chatter, no delegation state).

It is a **poor fit for live git repositories** for one reason: it has no
merge semantics. `rsync -a` from the dev host to a build host will
overwrite whatever is in the build host's checkout — including any
uncommitted state that originated *there* — with no three-way merge, no
conflict detection, and no history. If both hosts can ever legitimately
have independent state in the same tree (which is exactly the situation
`git worktree` isolation and this workspace's whole lane-branch discipline
exist to support), rsync will silently clobber one side. Git's own
transport (`clone`/`fetch`/`push`) is the right tool specifically because
it *is* a merge-aware distributed sync mechanism, not despite git being
"more complex than rsync."

Where rsync **is** the right call in this same workspace: syncing
read-only build caches/wheel caches to a build host (a real candidate for a
future addition here — `receive_artifact`'s own docstring already notes
artifact retrieval isn't fully wired yet), or the existing use of NFS for
genuinely shared, read-mostly data (see below) could in some cases be
replaced with a periodic rsync instead of a live mount, trading staleness
for eliminating the mount's live-protocol failure modes entirely.

## What NFS should keep being used for

Not every NFS mount in this workspace is wrong — only "a live git
repository shared as a mutable mount across hosts" is. Read-mostly,
genuinely-shared artifact data (package/wheel caches, model weights,
media libraries) is a legitimate NFS use case and out of scope for this
migration. The `agent-webui` pod's own NFS-mounted source volumes on R820
(`webui-deps`, `workspace`, `au-src`, `universal-skills-src`,
`agent-webui-src` — a live-reload/dev-mount pattern, not a build/CI one)
are a **separate, more disruptive migration**: they share the same NFS
client as the mounts this lane addresses, so fully eliminating the
pathology's root state requires touching them too, and that is exactly the
production-risk step this incident deliberately stopped short of and left
for explicit approval. A follow-up should evaluate moving that pod to a
baked image (matching this workspace's own documented lesson in
`webui-image-build-does-not-deploy-code`) or onto this same
`dispatch_build`-staged pattern, either of which would let R820's NFS
mounts to RW710 be removed entirely.

## Migrating a build/CI host

1. Pick (or create) a directory on the build host's own **local** disk —
   never under an NFS mount — to be its authorized worktree root, e.g.
   `/home/apps/build-worktrees` (what this lane used for R820).
2. `register_worker` that host: `host_id`, `inventory_alias` (the
   tunnel-manager/SSH-config alias — e.g. `"R820"`, which already resolves
   via `~/.ssh/config`), `repository_roots={"<repo>": "<local root>"}` per
   repository it builds, plus its measured/declared capacity.
3. Point CI/build dispatch at `dispatch_build(host_id=..., repository_id=...,
   origin=<dev host's canonical checkout over ssh, e.g.
   "genius@10.0.0.12:/home/apps/workspace/agent-packages/agents/<repo>">,
   tree_sha=<the exact commit to build>, command=[...])` instead of
   assuming the repo is already present via an NFS mount.
4. Leave the dev host's own canonical checkouts exactly as they are — this
   migration only changes how *other* hosts obtain code, never where the
   canonical checkout lives or how interactive development on the dev host
   itself works.
5. Do **not** unmount `/home/apps/workspace`/`/home/apps/worktrees` on a
   host that still has something depending on them (verify with `fuser -mv`
   and a CWD scan first, as this incident did) — R820's dev-only mounts were
   safe to unmount because nothing referenced them; a host running active
   interactive worktree sessions may not be.

## Open follow-ups (not done in this lane)

- Artifact/result retrieval from `dispatch_build` back to the caller is not
  wired (the action's own `note` field says so plainly) — only forward
  staging + remote command execution. `receive_artifact` exists but expects
  the caller to already have the bytes, not to pull them from the remote
  host.
- `RemoteWorkerRegistry`'s profile index (`host_id` → authorized roots /
  inventory alias) is **not** durably persisted the way
  `CapacityInventory`/`CapacityStore` is — a fresh process (a one-shot CLI
  invocation, or an MCP server restart) forgets registered profiles even
  though it remembers raw capacity. A long-lived MCP server process is
  unaffected (register once, dispatch many times, same process); a
  register-then-dispatch split across two separate CLI invocations is not.
  This is a real, disclosed gap surfaced while validating this lane, not
  something this lane's scope covers fixing.
- Forcing R820's wedged NFSv4 client to fully release its ~556K stuck
  delegations (`echo expire > /proc/fs/nfsd/clients/<id>/ctl` on RW710)
  remains undone, deliberately, because it touches state the live
  `agent-webui` pod's own NFS mounts share — see the session report for the
  full recommendation and required approval.
