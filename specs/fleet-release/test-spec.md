# RM-RELEASE-001 test contract

| Requirement | Fixture and test | Expected evidence |
|---|---|---|
| 01 | One Python, Rust, and JS package plus an omitted source; test census completeness. | Complete receipt only for full input; precise missing ID otherwise. |
| 02 | Acyclic four-stage graph, two-node SCC, upward phase edge, optional-only edge. | Stable order for first; explicit cycle/phase diagnostics; no silent edge loss. |
| 03 | Permuted equivalent input and one altered source revision. | Same digest for permutation, different for changed source. |
| 04 | Fake publisher returns ready, missing artifact, red CI, stale remote head. | Only ready predecessor admits dependent; all failures block mutation. |
| 05 | Drive Git, CLI, and MCP adapters against the same fake plan. | Same decision code, plan digest, and stage order across routes. |
| 06 | Fail between artifact publish and downstream readiness, then retry. | Partial receipt and safe reconciliation; no duplicate tag or false success. |

Run focused tests in `tests/test_workspace_release_dag.py`, `tests/test_workspace_release_plan.py`, `tests/test_phase_direction.py`, `tests/test_phased_push.py`, and `tests/test_phased_release_scope.py`. Add integration tests with temporary Git repositories and fake index/forge adapters. A contributor-controlled live rehearsal is supplementary, never a prerequisite to run ordinary PR checks. Preserve command output or CI URLs and exact commit SHA in the acceptance record.

Run `pre-commit run --all-files` where feasible, the repository's CCCC staged complexity gate (no new cyclomatic >10 or cognitive >15 function), and applicable repository quality checks. If jscpd or Dupehound is not configured here, record that gap and use focused duplication review rather than claiming those tools passed. KISS means one graph builder and one route admission path, no duplicated release authority.

## Documentation readiness identity admission (issue #4)

`tests/test_docs_readiness_fleet_action.py` exercises the narrow documentation
readiness boundary of RM-RELEASE-01: the caller's canonical workspace manifest owns
the expected agent identities, including shared skills and excluding agent tests.
A small offline fixture proves missing, extra, same-count substituted, and
duplicate identities fail before generation. Missing checkouts fail closed;
unlisted siblings and unrelated services cannot expand the fleet. This is not
evidence of dependency census completeness, release activation, or RM-RELEASE-R001 acceptance.
