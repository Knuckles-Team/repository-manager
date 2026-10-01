# RM-MATERIALIZE-001 test contract

| Requirement | Positive fixture | Negative fixture and expected result |
|---|---|---|
| 01 | Valid approval bound to exact proposal digest. | Changed path/base, expired proof, wrong actor or tenant: no worktree/ref. |
| 02 | Patch to one allowed text file in temporary repo. | Traversal, symlink, disallowed path, oversize, binary and stale base rejected before canonical mutation. |
| 03 | Deterministic validation profile and explicit two-file allowlist. | Red gate, post-validation tree drift, untracked file: no commit/queue. |
| 04 | Fake queue accepts candidate and records merged SHA. | Conflict, red hosted check, runner loss: receipt remains pending/failed. |
| 05 | Replay same request after response loss. | Altered replay fails; same replay yields same candidate and one commit. |
| 06 | Simulate unavailable approval authority, RM adapter, and receipt store. | No direct Git fallback; proposal stays pending. |

Implement unit tests for canonical serialization, bounded patch validation, state transitions, and error translation; integration tests use disposable Git repos and the existing queue fake/runner. Add a contract golden for `MaterializationReceipt/v1` and an independent Git SHA check. Acceptance captures exact main commit, full CI and quality result, queue record, and receipt digest.

Run repository pre-commit, Ruff/mypy/pytest, and CCCC (no new cyclomatic >10/cognitive >15 function). jscpd and Dupehound are not ordinary configured RM hooks; run where available and report missing tools honestly. KISS means one materializer service reusing existing worktree, gate, commit, and queue components, with no alternate Git publisher.
