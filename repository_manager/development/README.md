# Repository-development contracts

This package freezes the versioned data boundary used by the Repository Manager
development program.  It is intentionally additive and effect-free: importing
the models does not inspect a checkout, contact Graph-OS, resolve a
tunnel-manager host, execute a command, or create durable state.

## Authority and consumers

| Contract | Authority | First consumers |
| --- | --- | --- |
| `DevelopmentRequest` | MCP/CLI application service after validation | RMDD-02, RMDD-06, RMDD-20 |
| `RepositoryJobResult` and `LeaseRecord` | graph-os WorkItem projection | RMDD-02, RMDD-05, RMDD-06, RMDD-19 |
| `ResourceRequest` and `ResourceReservation` | scheduler admission | RMDD-07, RMDD-08, RMDD-09, RMDD-15 |
| `ExecutionCommand` and `ExecutionResult` | local/remote executor boundary | RMDD-07, RMDD-14, RMDD-15 |
| `BuildKey` and `BuildResult` | content-addressed build broker | RMDD-10, RMDD-12 |
| `ValidationEvidence` and `ValidationPolicy` | staged validation service | RMDD-11, RMDD-12, RMDD-13 |
| `Candidate`, `Generation`, and `CandidateVersion` | candidate/generation controller | RMDD-12, RMDD-13, RMDD-17 |
| `LaneReference` | durable lane registry | RMDD-09, RMDD-17, RMDD-20 |
| `TargetPolicy` and `RepositoryIdentity` | identity/configuration resolvers | RMDD-07, RMDD-14, RMDD-18 |
| `WorkspaceReleasePlan` and dependency records | workspace DAG/release planner | RMDD-18, RMDD-20, RMDD-24 |

Graph-os `WorkItem` remains the only durable job authority.  FastMCP task
operations and the CLI are projections/adapters; this package does not add a
second task store or scheduler.  Repository fragments, host inventory, and
workspace manifests retain their existing authorities.

## Version and compatibility

Every serialized model carries `contract_version: "1"` and rejects unknown
fields.  Additive fields require a new consumer-compatible release; changing
the meaning or validation of a persisted v1 field requires a new contract
version and an explicit migration decision.  `canonical_json` sorts mapping
keys, normalizes sets, and preserves declared sequence order.  `digest()` is a
full SHA-256 of that canonical payload and is suitable for idempotency/config
correlation.

The contract deliberately uses full Git object SHAs for immutable inputs and
named refs only for moving branch/base labels.  Absolute paths must already be
canonical and may be constrained by `configured_roots`; relative artifact and
changed-file paths cannot contain traversal components.

## Security and resource boundaries

`TargetPolicy` accepts only `local` or an authorized tunnel-manager inventory
alias.  Hostnames, usernames, passwords, key paths, proxy settings, and raw SSH
material are not model fields and are rejected as extras.  `ExecutionCommand`
contains fixed argv and bounded output/artifact limits, never a public shell
string.  `ResourceRequest` is an admission request; only a
`ResourceReservation` is evidence that capacity was actually reserved.

Certification evidence is tied to the exact generation ID, tree SHA, gate
configuration digest, command digest, host, and toolchain digest.  Stage-0
feedback is therefore not interchangeable with stage-2 certification.

Lifecycle transitions are explicit in `transitions.py`; terminal states have
no outgoing transition, retries are represented by a new WorkItem attempt, and
the model validators enforce the corresponding state/evidence combinations.

## Local execution consumer

The additive `repository_manager.execution` package consumes
`ExecutionCommand`/`ExecutionResult` without changing this contract boundary.
Its [local executor guide](../execution/README.md) documents fixed-argv
validation, authorized worktree roots, process-group cleanup, bounded redacted
logs, cancellation, heartbeat, and publication fencing.  Build, validation,
workspace, and remote lanes migrate their existing subprocess call sites to
that seam only after their own scheduler/transport policies are ready.
