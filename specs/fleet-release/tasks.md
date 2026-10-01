# RM-RELEASE-001 delivery tasks

- [x] Establish typed graph, plan, and frozen digest source in `development/workspace_release*.py` (source only, not yet merged; acceptance pending).
- [ ] Complete portable census and independent source-universe proof for every selected repository and package ecosystem.
- [ ] Close all dependency cycles or document an explicit supported scope that excludes them; prove no hidden edge omissions.
- [ ] Bind trusted artifact, forge, and repository evidence to every stage and retry.
- [ ] Route Git, CLI, and MCP through one fail-closed admission/executor, then retire the legacy path atomically.
- [ ] Add the positive/negative tests in `test-spec.md`, including failure/retry and stale-plan tests.
- [ ] Run full applicable gates and record exact-head CI, digest, source tree, and release rehearsal evidence in `spec.md`.
- [ ] Mark LANDED and ACCEPTED separately after merge and independent verification.
- [ ] Confirm every ID in [`requirements.md`](requirements.md) — `RM-RELEASE-R001` and `RM-RELEASE-01` through `RM-RELEASE-06` — is covered by the tasks above; file a follow-up task for any gap found during implementation.
