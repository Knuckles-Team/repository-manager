# RM-GOVERNANCE-001 architecture

## Existing components

`repository_manager/lane_registry.py` defines lane records and a fake durable authority; `native_lane_authority.py` is the production adapter. `worktree.py` creates and resolves worktrees; `safe_commit.py` performs gated commits; `merge_queue.py` and `merge_queue_runner.py` hold queue submission and drain behavior. `concept_coordination/` and `concept_actions.py` define reservation and verification actions. The CLI and MCP tools in `cli_commands/lane.py`, `cli_commands/merge_queue.py`, and `mcp_tools/lane.py` must invoke those same services. Extend existing services and remove duplicate authority at migration, rather than copying their internals.

## Authority and data flow

`request → validate repo identity/actor/base → lane authority allocate/fence → Git worktree → explicit patch/gates → gated commit receipt → queue admission → merge result → cleanup anchor/release`. Durable lane state records actor, repository, lease, state, commit/tree, and gate receipt. Git remains authority for refs and bytes. A concept reservation is bound to the candidate and checked again at commit and promotion; expiry is not authorization to reuse an unverified candidate.

## Migration and compatibility

Introduce an adapter that makes old callers use this repository's lane/queue/concept contracts. Route one entry point at a time, then delete the old producer once import searches and contract tests show no callers. Preserve stable CLI/MCP action names where possible; use typed error translation for changed failures. [`RM-CONNECTOR-R001`](../connector-boundary/spec.md) moves repository-manager's connector imports onto their legal package APIs after governance is available; it must not import an agent orchestration package merely to reach Git governance.

## Local setup

Clone this repository and run `uv sync --extra test`. Run `uv run pytest tests/test_lane_registry.py tests/test_native_lane_authority.py tests/test_safe_commit.py tests/test_merge_queue.py tests/test_merge_queue_runner.py`. Tests create temporary bare or ordinary Git repositories with disposable commits. For a manual rehearsal, create two local temporary repositories/worktrees under a directory you own and use the documented CLI; no hosted forge, live queue daemon, or shared workspace is required.

## Security and recovery

Resolve canonical paths and reject symlink escapes before any Git operation. Never issue `git stash` on a shared repository or stage all paths implicitly. A caller cannot bypass an authorization/fence failure by switching from CLI to MCP. Queue and cleanup actions must be idempotent; after a crash, replay from durable record plus Git ref state, not from an in-memory success flag. Preserve uncertain state until reconciliation proves it safe.
