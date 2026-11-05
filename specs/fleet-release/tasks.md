# RM-RELEASE — Tasks

One task per leaf row of `requirements.md`, producer first: census, graph and digest precede execution gating, routes and receipts; the rollout and the tag-only publication are independent. Each task names its row; the row holds the files, the caller chain and the proof.

## Tasks

- [ ] T001 [US1] RM-RELEASE-R001 Dependency-ordered fleet release planning
- [ ] T002 [US1] RM-RELEASE-R002 Complete per-repository dependency census
- [x] T003 [US1] RM-RELEASE-R003 Acyclic, typed dependency graph per release scope
- [x] T004 [US1] RM-RELEASE-R004 Immutable, content-addressed release plan digest
- [ ] T005 [US2] RM-RELEASE-R005 Dependency-ordered execution with trusted gating
- [ ] T006 [US2] RM-RELEASE-R006 One shared admission contract across release routes
- [ ] T007 [US2] RM-RELEASE-R007 Per-stage receipts with retry and rollback evidence
- [x] T008 [P] [US2] RM-RELEASE-R008 Publish this package only from version tags
- [x] T009 [P] [US3] RM-RELEASE-R009 Deterministic, resumable dependency-floor rollout through PRs
