# EH-349 delivery tasks

- [ ] Define versioned proposal, approval proof, receipt, and typed failure schemas with canonical digest goldens.
- [ ] Implement authenticated admission and path/base/size validation before any worktree effect.
- [ ] Compose existing lane, worktree, validation, explicit commit, and merge queue services in one materializer.
- [ ] Add idempotent state persistence and uncertain-delivery reconciliation.
- [ ] Expose one CLI/MCP application operation and migrate callers; delete direct Git fallback routes.
- [ ] Add all positive, negative, failure injection, and temporary Git integration tests in `test-spec.md`.
- [ ] Run exact-head full gates and queue rehearsal; store commit/CI/receipt evidence before LANDED or ACCEPTED labels.
