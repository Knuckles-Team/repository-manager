# RM-GOVERNANCE delivery tasks

- [x] Build the repository-manager governance APIs as isolated source (not yet merged to the default branch; merged-head verification pending).
- [ ] Wire all production CLI, MCP, safe-commit, merge-queue, and concept callers to the RM authority.
- [ ] Remove active duplicate governance code and obsolete imports from the former owner; complete the [`RM-CONNECTOR-R001`](../connector-boundary/spec.md) repository-manager import migration separately.
- [ ] Add conflict, stale-fence, crash recovery, explicit-stage, and safe-prune tests from `test-spec.md`.
- [ ] Run fixture-based entry-point rehearsal and full applicable quality gates at the exact candidate commit.
- [ ] Merge to main; record source commit, CI, import scan, and queue/recovery evidence; mark LANDED then ACCEPTED only on proof.
- [ ] Lease heavy gates on counted slots of inventory hosts and refuse when none is free. Closes RM-GOVERNANCE-R023.
- [ ] Confirm every ID in [`requirements.md`](requirements.md) — `RM-GOVERNANCE-R001`, `RM-GOVERNANCE-R024` through `RM-GOVERNANCE-R030`, and `RM-GOVERNANCE-R002` through `RM-GOVERNANCE-R022` — is covered by the tasks above; file a follow-up task for any gap found during implementation.
