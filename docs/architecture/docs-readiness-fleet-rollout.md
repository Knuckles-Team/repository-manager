# Documentation-readiness fleet rollout authority

`repository_manager.docs_readiness_rollout` owns the selection and transaction
boundary for NE-146. It is intentionally narrower than the generator and the
Pages/TCK implementation:

- `universal-skills` remains the only artifact generator (NE-137).
- `pipelines` remains the Pages/TCK authority (NE-144).
- Repository Manager selects the exact fleet, plans bounded waves, proves each
  target checkout is safe, records revision/digest evidence, and reconciles an
  operator-approved transaction.

No worker invocation edits the 75 repositories. The manifest is a checked-in
selection authority, not a command to clone, branch, push, or regenerate the
fleet.

## Exact fleet and source findings

`repository_manager/docs_readiness_fleet_manifest.json` contains exactly 75
publishable identities below `agent-packages/`. `agent-packages/agents/tests`
is explicitly excluded as a fixture. No filesystem sibling is promoted into
the selection, and a missing checkout or cardinality change fails before an
adapter is called.

The source audit treats a Pages surface as either a `pages` job or an active
call to the reviewed reusable Pages workflow. This matters because the three
shared/core projects folded Pages into `advisory.yml` rather than retaining a
standalone `pages.yml`. The current source findings are:

- Missing Pages surfaces (2): `agents/ciso-assistant-api` and
  `agents/onetrust-api`.
- Missing `site_url` declarations (4): `agent-utilities`, `agents/mealie-mcp`,
  `agents/microsoft-agent`, and `agents/vector-mcp`.

`audit_source_findings()` re-reads those facts from the supplied workspace and
refuses source drift. It never writes the missing workflows or URLs; the
actual generated fleet rollout remains a separate, operator-approved step.

## Deterministic waves

Projects are sorted by their manifest-relative identity and grouped by the
explicit wave field. The current plan is:

| Wave | Scope | Count | Dependency |
| --- | --- | ---: | --- |
| 1 | foundations (shared utilities, frontends, engine, skills) | 7 | none |
| 2 | early connectors | 14 | wave 1 |
| 3 | core connectors | 19 | wave 2 |
| 4 | platform connectors | 17 | wave 3 |
| 5 | tail connectors | 18 | wave 4 |

The planner refuses duplicate/unknown selections, empty waves, count drift,
or a partial default fleet. A manually supplied one-project selection is still
checked against the same manifest and cannot escape its identity boundary.

## Safety and evidence contract

Before preview/apply/rollback, Repository Manager resolves the identity beneath
the supplied workspace root, rejects symlink/path escapes, and records only:

- exact `HEAD` revision;
- current branch name and clean/dirty boolean;
- linked-worktree count; and
- a SHA-256 digest over bounded documentation/configuration inputs.

Apply requires a clean checkout on `main` with exactly one worktree. Dirty trees,
detached/non-main branches, missing repositories, and linked worktrees refuse
before generator invocation. No dirty path, repository content, exception text,
credential, or host path enters the journal response.

The adapter must identify the exact generator revision, version, and
`agent-readiness/v1` schema plus the exact Pages/TCK revision and
`pages-readiness-tck/v1` schema. The caller must provide full immutable
revisions for both NE-137 (generator) and NE-144 (Pages/TCK); `main`, short
hashes, or missing dependency evidence are rejected. The manifest's source
authority and surface-policy fields are schema-bound and included in its
content digest, so a policy-only edit cannot silently reuse an old plan.

Generator results are reduced to bounded output names, version/schema, and a
64-hex provenance digest. Absolute paths, traversal, duplicate outputs, raw
exceptions, and malformed evidence are rejected.

## Transaction semantics

`TransactionJournal` writes a bounded JSON document atomically and fsyncs it
before any apply adapter call. An apply record progresses:

```text
prepared -> applied
         \-> rolled_back
         \-> rollback_failed (operator recovery required)
```

The operation key binds manifest digest, project identity, source revision and
digest, generator revision, and TCK revision. Exact retries return
`replayed: true`; a prepared or rollback-failed transaction refuses rather than
guessing; a different active operation for the same identity is a conflict.
Apply verifies the canonical output digest after publication and invokes the
adapter's bounded rollback seam on any publication/verification failure.
Rollback requires explicit confirmation and the original source revision/digest
to remain unchanged. It only delegates restoration of the generator-owned
artifact namespace and is itself replay-safe.

Preview is read-only and must use the adapter's staging path. It never creates
branches, worktrees, commits, pushes, or edits a target repository.

## Root-only execution gates

Workers may run `git diff --check` only. After NE-137 and NE-144 are reviewed and
landed, root should run the focused rollout suite, render a real exact-75
preview with full dependency revisions, verify source findings, and inspect
the journal/revision/digest evidence. Only root may authorize per-wave adapter
execution, generated changes, commits, pushes, Pages TCKs, live deployment, or
rollback.
