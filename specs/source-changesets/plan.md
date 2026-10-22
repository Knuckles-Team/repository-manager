# RM-SCM — Design and implementation plan

Governing spec: [spec.md](spec.md).

Component: Source control
Depends-on: [EG-SCM](https://github.com/Knuckles-Team/epistemic-graph/blob/main/specs/source-changesets/spec.md), [SDK-SCM](https://github.com/Knuckles-Team/agent-connector-sdk/blob/main/specs/SDK-SCM/spec.md)
References: [ARBORA-OPS](https://github.com/Knuckles-Team/arbora/blob/main/specs/hosted-api-operations/spec.md), [PIPE-GATES](https://github.com/Knuckles-Team/pipelines/blob/main/specs/gates/spec.md)

## Existing system and reuse

`repository_manager/repository_manager.py` already clones, pulls and pushes repositories. RM-MATERIALIZE writes one governed proposal to git. The SDK source client of SDK-SCM-R001 reaches the served `source.*` operations. This spec adds a `repository_manager/source_changesets/` package over those parts.

## Import

The import walks git commits in topological order and submits one changeset per commit. The tree hash of each changeset must equal the commit tree hash (EG-SCM-R010). A re-run imports only new commits.

## Mirror sync

After each `source.land`, the sync exports the commit and pushes it to the public and internal git mirrors. A commit made on a mirror comes back through the import. A diverged mirror stops the sync. The sync never force-pushes.

## Development tools

One connector tool serves each `source.*` operation. arbora serves these tools inside its MCP surface. The agent loop of AU-SCM-R001 edits through them.

## Wave step

PIPE-GATES-R006 orders the repositories into seven waves. The wave step moves one repository to engine-primary, or back to git-primary.
