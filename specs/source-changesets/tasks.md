# RM-SCM — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

One task per leaf row, producer first: models, behavior, wiring, refusal, evidence, hot-path proof. Rollup rows are covered by their children.

## Tasks

- [ ] T001 [US1] RM-SCM-R001.1 Define the models of the git import
- [ ] T002 [US1] RM-SCM-R001.2 Implement the behavior of the git import
- [ ] T003 [US1] RM-SCM-R001.3 Wire the git import
- [ ] T004 [US1] RM-SCM-R001.4 Add the typed refusals of the git import
- [ ] T005 [P] [US1] RM-SCM-R001.5 Write the receipts of the git import
- [ ] T006 [US1] RM-SCM-R001.6 Prove the hot path of the git import
- [ ] T007 [US2] RM-SCM-R002.1 Define the models of the mirror sync
- [ ] T008 [US2] RM-SCM-R002.2 Implement the behavior of the mirror sync
- [ ] T009 [US2] RM-SCM-R002.3 Wire the mirror sync
- [ ] T010 [US2] RM-SCM-R002.4 Add the typed refusals of the mirror sync
- [ ] T011 [P] [US2] RM-SCM-R002.5 Write the receipts of the mirror sync
- [ ] T012 [US2] RM-SCM-R002.6 Prove the hot path of the mirror sync
- [ ] T013 [US3] RM-SCM-R003.1 Register the eight rm_source tools
- [ ] T014 [US3] RM-SCM-R003.2 Add the typed refusals of the served source tools
- [ ] T015 [US3] RM-SCM-R003.3 Prove the hot path of the served source tools
- [ ] T016 [US4] RM-SCM-R004.1 Implement the behavior of the wave step
- [ ] T017 [US4] RM-SCM-R004.2 Add the typed refusals of the wave step
- [ ] T018 [US4] RM-SCM-R004.3 Write the receipts of the wave step
