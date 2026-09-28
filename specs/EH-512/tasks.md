# EH-512 delivery tasks

- [x] Build isolated source carrier for repository-manager governance APIs (source only; merged-head verification pending).
- [ ] Wire all production CLI, MCP, safe-commit, merge-queue, and concept callers to the RM authority.
- [ ] Remove active duplicate governance code and obsolete imports from the former owner; complete EH-483's repository-manager import migration separately.
- [ ] Add conflict, stale-fence, crash recovery, explicit-stage, and safe-prune tests from `test-spec.md`.
- [ ] Run fixture-based entry-point rehearsal and full applicable quality gates at the exact candidate commit.
- [ ] Merge to main; record source commit, CI, import scan, and queue/recovery evidence; mark LANDED then ACCEPTED only on proof.
