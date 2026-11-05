# RM-GOVERNANCE — Tasks

One task per leaf row of `requirements.md`, producer first: the single authority and the lane, worktree, commit and queue rows precede the cleanup, hook, gate and host-lease rows. Each task names its row; the row holds the files, the caller chain and the proof. The host-lease rows run in series because they edit `repository_manager/governance/lanes.py`.

## Tasks

- [x] T001 [US1] RM-GOVERNANCE-R001 repository-manager hosts the Git development-governance modules
- [x] T002 [US1] RM-GOVERNANCE-R024 Single authority for Git development governance
- [x] T003 [US1] RM-GOVERNANCE-R025 Branch allocation tracks ownership, lease, and lifecycle state
- [x] T004 [US1] RM-GOVERNANCE-R026 Isolated checkouts never corrupt shared Git config
- [x] T005 [US1] RM-GOVERNANCE-R027 Commits stage only reviewed paths and verified gate identity
- [x] T006 [US2] RM-GOVERNANCE-R028 Merge-queue promotion rechecks identity before reporting success
- [x] T007 [US2] RM-GOVERNANCE-R029 Cleanup proves containment before deleting a branch checkout
- [x] T008 [US3] RM-GOVERNANCE-R030 Concept reservations are scoped and cannot be silently reused
- [x] T009 [US3] RM-GOVERNANCE-R031 Export the canonical development lane hook
- [x] T010 [US3] RM-GOVERNANCE-R032 The package runs on Windows as well as POSIX
- [ ] T011 [US4] RM-GOVERNANCE-R002 readme-deploy-block hook must not silently skip
- [ ] T012 [US2] RM-GOVERNANCE-R003 Cleanup removes isolated checkouts and scratch only after merge
- [x] T013 [US4] RM-GOVERNANCE-R004 Workspace check refuses credential-embedded remote URLs
- [ ] T014 [US4] RM-GOVERNANCE-R005 Complexity gate scores exhaustive-dispatch functions correctly
- [ ] T015 [US4] RM-GOVERNANCE-R006 Clone-duplication gate enforces zero new pairs before push
- [ ] T016 [US2] RM-GOVERNANCE-R007 Routine cleanup prunes merged branches and build caches
- [ ] T017 [US4] RM-GOVERNANCE-R008 KISS hook resolves test module path declarations correctly
- [ ] T018 [US4] RM-GOVERNANCE-R009 Duplication gate fails closed on an empty-change commit
- [ ] T019 [US4] RM-GOVERNANCE-R010 Reviewed-distinct clone pairs are tracked in a registry
- [ ] T020 [US4] RM-GOVERNANCE-R011 Status page stays in sync with capability and reservation docs
- [ ] T021 [US4] RM-GOVERNANCE-R012 Build verification on extracted trees uses content hashes
- [x] T022 [US4] RM-GOVERNANCE-R013 A fast rebase must not silently skip the hook suite
- [ ] T023 [US4] RM-GOVERNANCE-R014 Completion claims require full hook-run logs as proof
- [ ] T024 [US4] RM-GOVERNANCE-R015 Continuous review-and-merge replaces batch integration
- [ ] T025 [US2] RM-GOVERNANCE-R016 Cleanup removes leftover backup references and build scratch
- [x] T026 [US4] RM-GOVERNANCE-R017 Build submission is asynchronous, not polling-based
- [x] T027 [US4] RM-GOVERNANCE-R018 Dependency-readiness hook works from any checkout location
- [x] T028 [US4] RM-GOVERNANCE-R019 A fix is accepted only when all quality gates pass together
- [ ] T029 [US4] RM-GOVERNANCE-R020 Duplication gate distinguishes real new duplication from moves
- [ ] T030 [US4] RM-GOVERNANCE-R021 Secret-history exceptions are digest-pinned and self-checking
- [x] T031 [US4] RM-GOVERNANCE-R022 The repository passes its own full local gate suite
- [ ] T032 [US4] RM-GOVERNANCE-R023.1 Typed host lease
- [ ] T033 [US4] RM-GOVERNANCE-R023.2 Read capacity from inventory
- [ ] T034 [US4] RM-GOVERNANCE-R023.3 Lease a slot per host
- [ ] T035 [US4] RM-GOVERNANCE-R023.4 Refuse when no host has headroom
- [ ] T036 [US4] RM-GOVERNANCE-R023.5 Release on crash
- [ ] T037 [US4] RM-GOVERNANCE-R023.6 Wire into the gate runner
