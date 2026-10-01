# RM-CONNECTOR-001 delivery tasks

- [ ] Produce reviewed import inventory and stable tool schema baseline.
- [ ] Complete the [`RM-GOVERNANCE-R001`](../development-governance/spec.md) governance cut and route all RM governance callers locally.
- [ ] Migrate connector transport to the SDK public interface and retain only legitimate orchestration imports.
- [ ] Remove private, dead, and fallback adapters after tests prove no caller remains.
- [ ] Add clean-install, import-direction, MCP/CLI contract, and negative tests in `test-spec.md`.
- [ ] Run exact-head quality and package gates, merge, and record commit/CI evidence before marking LANDED or ACCEPTED.
- [ ] Confirm every ID in [`requirements.md`](requirements.md) — `RM-CONNECTOR-R001` and `RM-CONNECTOR-01` through `RM-CONNECTOR-05` — is covered by the tasks above; file a follow-up task for any gap found during implementation.
