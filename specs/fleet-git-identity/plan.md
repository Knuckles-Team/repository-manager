# RM-IDENTITY-001 architecture

## Reuse and boundaries

Reuse `repository_manager/worktree.py` for safe repository identity and isolation, `merge_queue.py` for pause/drain coordination, `workspace_manifest.py` for portable repository selection, and `development/workspace_release.py` for dependency ordered coordination and digest patterns. The history rewrite engine is a new bounded application service only after owner approval; CLI and MCP are adapters to that one service. Git objects and refs remain authoritative; receipts index what was proven.

## Proposed sequence

`discover exact refs/remotes → validate public identity policy → deterministic dry run → approval bound to preview digest → pause queues → write backup refs/object map → rewrite into quarantine namespace → independently verify trees/graph/tags → atomic local ref transaction → lease-protected remote publication → reconcile each remote/PR → resume queues`. Every phase has a durable receipt and an idempotent retry key. A changed source ref restarts preview and approval; it is never silently rebased.

## Data model

`IdentityPolicy/v1`: exact aliases, canonical human/Claude/Codex identities, unknown-identity rule, and digest. `HistoryPreview/v1`: repository IDs, old ref targets, proposed new ref targets, object mapping digest, signed-object inventory, expected tree parity, remotes, and expiry. `HistoryRewriteReceipt/v1`: approval/preview digests, backup ref namespace, object-map digest, verification result, local transaction outcome, per-remote result, queue pause/resume state, and recovery instructions. Receipts must omit credentials and private filesystem paths.

## Failure and recovery

Before local ref update, remove only quarantine refs after preserving a diagnostic receipt. After local update but before remote success, retain backups and pause queues; reconcile each remote against old/new expected SHA. Never blanket force-push or delete backup refs in the same change. Recovery replays exact old SHA from backup refs with leases, only after another reviewed approval. Unknown authors and unhandled signatures refuse execution.

## Contributor setup

Clone this repository, run `uv sync --extra test`, and use temporary Git repositories with commits by synthetic users, branches, annotated tags, and a second local bare remote. Preview and failure tests use only those fixtures. A real fleet run is an operational event separate from code acceptance.
