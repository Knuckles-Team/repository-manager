# RM-IDENTITY-001 test contract

| Requirement | Positive case | Negative/recovery case |
|---|---|---|
| 01 | Enumerate branches, tags, notes, and alternate refs in a temporary repository. | Hidden or unreadable ref and divergent remote refuse run. |
| 02 | Two dry runs produce identical preview digest and object map. | Source mutation between preview and execution invalidates approval. |
| 03 | Valid approval over exact repository set and policy. | Expired/wrong-actor/changed-policy approval leaves refs unchanged. |
| 04 | Multiple parents, merges, author vs committer aliases, annotated tags. | Signed objects and unknown authors refuse until decisions are recorded; every old/new tree digest matches. |
| 05 | Backup refs, quarantine, verification, atomic ref update. | Failure at every phase preserves old refs or a complete rollback anchor. |
| 06 | Two local bare remotes receive lease updates. | One remote drifts or rejects; receipt says PARTIAL and retry does not overwrite drift. |
| 07 | Queue pause/restart around a rewritten candidate. | Candidate based on old SHA is refused after cutover; open PR branch recovery is documented. |

Add property tests for object mapping determinism and collision handling. Use a separate verifier process or implementation to compare old/new trees and commit graph; the rewrite engine cannot grade its own output alone. Run the repository's pre-commit suite and CCCC gate, plus Ruff/mypy/pytest. jscpd and Dupehound are supplemental if available; report the actual command/results rather than inventing a pass. KISS requires one policy evaluator and one Git rewrite transaction service, not per-remote duplicated logic.
