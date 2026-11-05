# RM-CONNECTOR — Tasks

One task per row of `requirements.md`, in dependency order: governance first, connector transport second, the inventory and contract golden next, and removal of dead adapters last. Each task names its row; the row holds the files, the caller chain and the proof.

## Tasks

- [x] T001 [US2] RM-CONNECTOR-R004 Keep governance imports under repository-manager authority
- [x] T002 [US2] RM-CONNECTOR-R003 Route connector transport imports through the public SDK
- [x] T003 [US1] RM-CONNECTOR-R001 Migrate connector imports to the public SDK boundary
- [ ] T004 [US2] RM-CONNECTOR-R002 Classify every import from the orchestration package
- [x] T005 [US1] RM-CONNECTOR-R005 Preserve MCP/CLI contract during connector migration
- [ ] T006 [US2] RM-CONNECTOR-R006 Remove dead connector adapters after cutover
