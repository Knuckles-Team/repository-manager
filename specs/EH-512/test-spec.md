# EH-512 test contract

| Requirement | Positive case | Negative/recovery case |
|---|---|---|
| 01 | CLI and MCP dispatch to the same lane/queue service. | Import scanner detects any active duplicate governance implementation. |
| 02 | Allocate, heartbeat, transition, complete one lane. | Conflicting repository/path claim and stale fence reject with unchanged state. |
| 03 | Two temporary worktrees commit independently. | Failed creation leaves shared Git config and first worktree usable. |
| 04 | Explicit two-path commit records exact tree/gate receipt. | Extra untracked path, red gate, or modified index prevents admission. |
| 05 | Queue merges a clean candidate into a new base. | Stale base, merge conflict, and killed runner yield a retryable record. |
| 06 | Merged clean branch prunes with recovery proof. | Dirty or unmerged branch remains with a reason and anchor. |
| 07 | Reserve, verify, materialize, release a concept. | Other actor, expired fence, or mismatched candidate cannot reuse claim. |

Place tests in the existing `tests/test_lane_registry.py`, `tests/test_native_lane_authority.py`, `tests/test_safe_commit.py`, `tests/test_merge_queue.py`, `tests/test_merge_queue_runner.py`, and concept coordination tests. Use disposable Git fixtures and deterministic fake clock/authority adapters; optional forge tests are additive. Save exact commit/CI receipts and an import inventory when accepting.

Run `pre-commit run --all-files`, applicable CCCC staged complexity check (no new cyclomatic >10/cognitive >15), Ruff/mypy/pytest, and duplication review. jscpd and Dupehound are not configured by this repository's ordinary hook; record their availability instead of asserting a pass. KISS requires one lane registry, one queue, and one concept claim authority, with adapters rather than duplicate implementations.
