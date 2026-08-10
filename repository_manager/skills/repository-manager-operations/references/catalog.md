# Provider workflow catalog

Load only the workflow relevant to the current request. Every entry below is a
`repository-manager` skill directory under this package (`repository_manager/skills/`);
this catalog only points to them — skill discovery itself walks the package for every
`SKILL.md`, so a skill also works if this catalog drifts, but keep it current anyway.

## Start here
- [repository-manager-development-lifecycle](../../repository-manager-development-lifecycle/SKILL.md): The default entrypoint for one unit of work — plan, start, work with a live heartbeat, check, submit, watch status, and abort — composed entirely from the repository-manager MCP/CLI surface. Use this FIRST for any repository change an agent will land.

## Lane / worktree lifecycle
- [repository-manager-lane-lifecycle](../../repository-manager-lane-lifecycle/SKILL.md): Run one unit of work as an isolated lane in a repository many other lanes are editing at the same time — opening, staying isolated, diagnosing, and closing out.
- [repository-manager-worktree-orchestration](../../repository-manager-worktree-orchestration/WORKFLOW.md): Concurrent multi-session git worktree orchestration — create, list, sync, merge, prune, and audit linked worktrees under a shared workspace.
- [repository-manager-fleet-scale-operations](../../repository-manager-fleet-scale-operations/SKILL.md): Development across MANY repositories and many concurrent lanes at once — bulk worktree creation, workspace-wide audits, safe mass pruning, draining several merge queues, and sizing concurrency against disk I/O and swap.

## Landing and certification
- [repository-manager-merge-and-reconcile](../../repository-manager-merge-and-reconcile/SKILL.md): Land a branch into a shared base dozens of other lanes are landing into at the same time, and reconcile it when it conflicts — the serialized merge queue, differential gating, and the conflict-resolution decision procedure.
- [repository-manager-candidate-certification](../../repository-manager-candidate-certification/SKILL.md): The Candidate/Generation certification vocabulary and the `rm_concepts` marker-verification actions, before trusting a merge-queue outcome as landable.

## Git and build
- [repository-manager-bulk-git-operations](../../repository-manager-bulk-git-operations/WORKFLOW.md): Bulk, parallel git operations across a whole workspace of repositories — clone, pull, push, add, commit, pre-commit, phased push, and enumerate every repo across a GitLab instance / GitHub org into an ingest manifest. Raw host git commands are permanently retired and always refused.
- [repository-manager-build-coordination](../../repository-manager-build-coordination/SKILL.md): Content-addressed build/cache coordination for any repository — dedup-or-build requests, cache-key status, published artifacts, why a key missed cache, and bounded cache reclamation.

## Validation and release
- [repository-manager-workspace-validation](../../repository-manager-workspace-validation/WORKFLOW.md): Install, build, validate, and version-maintain the managed projects of a workspace — pre-commit + pytest validation per project, install/build ecosystems, phased version bumps and maintenance.
- [repository-manager-workspace-release](../../repository-manager-workspace-release/SKILL.md): Preview the topologically-phased version-bump plan, drive validation with an explicitly consented chain into version bump and phased push, and manage the `workspace.yml` manifest.

## Concepts and remote workers
- [repository-manager-concept-coordination](../../repository-manager-concept-coordination/SKILL.md): Reserve, inspect, release, and materialize a CONCEPT:ID claim, and verify a candidate's/generation's introduced concept markers. ⚠ Every mutating action refuses today (`ConceptAuthorityUnavailable`) — read before assuming this allocates anything.
- [repository-manager-worker-operations](../../repository-manager-worker-operations/SKILL.md): Register a remote host's weighted capacity, run the dispatch-time entitlement recheck, stage/verify an immutable source commit, stream artifacts/logs, and reconcile a lost host. ⚠ `recheck` without tunnel-manager and `host_loss_reconcile` refuse honestly today.
