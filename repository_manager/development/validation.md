# RMDD-11 staged validation

`repository_manager.validation` is the policy/evidence/runner seam for the five
validation stages.  It does not land branches, push, create a second task
store, or compile outside the common executor and scheduler.

## Stages

| Stage | Input tree | Purpose | Certification effect |
| --- | --- | --- | --- |
| `feedback` | lane commit snapshot | format, focused lint/tests, and config feedback | never landable |
| `integration` | sealed synthetic generation | merge reconciliation and differential fast gates | blocks the generation |
| `certification` | the same immutable generation SHA | full pre-commit/suite/build/security gates | only this stage can issue a certificate |
| `smoke` | the locally landed target | startup, registration, wiring, and artifact health | blocks downstream release |
| `release` | frozen workspace DAG | cross-project validation and release readiness | hands off to workspace mutation, bump, and push services |

Profiles contain fixed argv, path selectors, blocking/advisory/deferred mode,
weighted resource estimates, timeout policy, baseline mode, and artifact
dependencies.  Built-in families are `docs`, `python`, `rust`, `frontend`,
`schema`, `concept`, `deployment`, and `release`.  A repository may replace a
family with its validated versioned `.mergequeue.yaml`; unknown safety-relevant
keys are refused by `config_schema`, never ignored.

## Exact evidence

Every `GateEvidence` names the exact tree SHA, generation (when applicable),
profile/config digest, command digest, toolchain digest, target host, resource
policy digest, baseline identity, job/dependency IDs, bounded output, and
artifact/log references.  Differential mode refuses an unreadable baseline;
it never treats an unavailable base as an empty failure set.  Baseline cache
keys include base SHA, gate/config, command, toolchain, and host, so any of
those changes invalidate a cached observation.

`ValidationCertificate.issue` accepts only certification evidence for one exact
generation/tree and verifies all blocking gates.  The verifier rejects stage-0
or stage-1 evidence, missing/failed/deferred blocking evidence, digest drift,
host/toolchain/resource drift, duplicate gates, and certificate/evidence set
mismatches.  A failed verification is explicitly *not a certificate*.

## Runner boundary

`ValidationRunner.plan` seals a deterministic dependency-linked job DAG.  Each
job carries the immutable tree/worktree path, gate command, dependency IDs,
configuration/toolchain/resource digests, base SHA, and generation.  `submit`
requires the graph-os WorkItem adapter.  `run` admits each job before invoking
the fixed-argv `ValidationExecutor`, releases its reservation in all result,
exception, timeout, and cancellation paths, and classifies code, environment,
resource, cancellation, timeout, stale-tree/fence, baseline, dependency, and
reconciliation outcomes separately.

For a feedback or integration request pointed at a dirty lane tree, the runner
calls RMDD-26 `safe_commit` to stage the complete tree (including deletions and
untracked files), then gates the resulting committed SHA.  When a repository
has a pre-commit configuration, `safe_commit(defer_gate=True)` creates only a
WIP snapshot with `--no-verify` and returns `gate_deferred=True`; that snapshot
is not evidence or certification.  Only the selected gate(s) from the
resolved profile are then submitted, resource-admitted, and executed through
the common fixed-argv executor against the committed SHA; a profile that does
not select a pre-commit gate does not claim that the deferred hook ran.  The
run and each resulting evidence record retain `snapshot_gate_deferred=True`.
Certification additionally requires a `snapshot_gate_replayed=True` record,
which is set only when a selected gate whose fixed argv is explicitly a
pre-commit command passes under the executor.  Without that proof, a
certification certificate is refused even if lightweight gates passed.  A
dirty tree without that snapshot path, a symlink-ambiguous worktree, a moved
HEAD, or a status that cannot be read is refused before submission.  The
stage-0/1 path therefore does not reproduce the pre-commit unstaged-files
window or run a heavy hook outside the scheduler/executor boundary.

The default resource adapter is fail-closed until RMDD-27 native reservation
authority is installed.  `LocalTestAdmission` and
`FakeValidationJobAuthority` are test fixtures only and must not be described
as cross-host authority.  Post-land smoke and release helpers return typed
handoffs; they do not perform landing, workspace mutation, version bumps, or
pushes.  This lane currently submits durable-shaped jobs through injected
adapters only; it has no production claim/terminal WorkItem adapter or executor
fence heartbeat.  It is not a production-execution implementation until the
RMDD-27/RMDD-29 authority and heartbeat binding is closed.
