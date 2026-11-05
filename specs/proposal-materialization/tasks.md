# RM-MATERIALIZE — Tasks

One task per leaf row of `requirements.md`, producer first: admission precedes the worktree, the patch, the commit and the queue. Each task names its row; the row holds the files, the caller chain and the proof.

## Tasks

- [ ] T001 [US2] RM-MATERIALIZE-R002 Verify the approval binding and the approval authority before any worktree
- [ ] T002 [US2] RM-MATERIALIZE-R003 Create the isolated worktree at the approved base and apply only the allowlisted patch
- [ ] T003 [US1] RM-MATERIALIZE-R004 Commit the explicit path list after the declared gate
- [ ] T004 [US1] RM-MATERIALIZE-R005 Submit to the merge queue and reconcile the landing
- [ ] T005 [US3] RM-MATERIALIZE-R006 Persist the immutable receipt and replay idempotently
- [ ] T006 [US3] RM-MATERIALIZE-R007 Keep the request pending when a dependency is unavailable
- [ ] T007 [US1] RM-MATERIALIZE-R001 Expose the materializer through one CLI verb and one MCP tool
