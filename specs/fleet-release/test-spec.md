# RM-RELEASE — Test specification

Proof level: focused tests with offline fixtures, temporary Git repositories and fake index, forge and runner adapters. A contributor-controlled live rehearsal is supplementary, never a prerequisite for ordinary PR checks.

## Bound tests

- Test: `tests/test_fleet_release_gate.py::test_incomplete_census_refuses_before_any_live_release_authority` and `tests/test_fleet_evidence_provider.py::test_current_manifest_without_verified_metadata_refuses`. *(RM-RELEASE-R001)*
- Test: `tests/test_fleet_census.py::test_bounded_census_captures_four_evidence_families`, `tests/test_fleet_source_universe.py::test_extra_tracked_source_refuses_manifest_self_attestation` and `tests/test_docs_readiness_fleet_action.py::test_manifest_selection_drift_fails_closed_before_generator`. *(RM-RELEASE-R002)*
- Test: `tests/test_workspace_release_dag.py::test_cycle_refuses_before_a_graph_is_returned`, `tests/test_fleet_order_receipt.py::test_fleet_scc_census_reports_each_exact_component` and `tests/test_phase_direction.py::test_phase4_runtime_dependency_on_phase5_package_fails`. *(RM-RELEASE-R003)*
- Test: `tests/test_workspace_release_plan.py::test_plan_digest_and_stages_are_independent_of_input_order`, `tests/test_workspace_release_plan.py::test_profile_source_and_consent_changes_change_exact_digest` and `tests/test_fleet_order_receipt.py::test_fleet_plan_binding_rejects_stale_or_tampered_order`. *(RM-RELEASE-R004)*
- Test: `tests/test_dependency_readiness.py::test_wheel_without_sdist_passes_with_the_sdist_missing_advisory`, `tests/test_dependency_readiness.py::test_sdist_only_version_fails`, `tests/test_dependency_readiness.py::test_sdist_only_version_on_the_allow_list_passes`, `tests/test_dependency_readiness.py::test_in_progress_release_run_reports_partial_publish` and `tests/test_dependency_readiness.py::test_concluded_run_with_a_wheel_passes_without_asking_the_forge`. The stage-blocking proof `tests/test_fleet_release_gate.py::test_stale_predecessor_blocks_dependent_stage` is planned. *(RM-RELEASE-R005)*
- Test: `tests/test_fleet_release_gate.py::test_open_cutover_refuses_legacy_git_routes_before_effect` and `tests/test_fleet_release_gate.py::test_closed_cutover_preserves_legacy_route_selection`. *(RM-RELEASE-R006)*
- Test: `tests/test_repository_job_service.py::test_reconcile_enqueues_idempotent_follow_up_without_inline_mutation`, `tests/test_repository_job_service.py::test_retry_chain_consumes_the_original_budget_without_replenishing_it` and `tests/test_durable_build_broker.py::test_worker_surfaces_release_reconciliation_without_rerunning`. *(RM-RELEASE-R007)*
- Test: `tests/test_package_publication_workflow.py::test_source_ci_still_validates_main_and_pull_requests` and `tests/test_package_publication_workflow.py::test_connector_workflow_is_the_only_source_ci_caller`. *(RM-RELEASE-R008)*
- Test: `tests/test_rollout.py::test_apply_opens_one_pr_with_bump_changelog_and_status_regen`, `tests/test_rollout.py::test_resume_is_idempotent_no_duplicate_pr_or_double_bump`, `tests/test_rollout.py::test_lock_that_misses_the_floor_fails_the_row`, `tests/test_rollout.py::test_release_tags_only_after_merge_and_floor_met`, `tests/test_rollout.py::test_refuses_claude_identity` and `tests/test_rollout.py::test_default_manifest_path_is_arbora`. *(RM-RELEASE-R009)*

## Planned

- `tests/test_fleet_release_gate.py::test_stale_predecessor_blocks_dependent_stage` (planned). A fake publisher returns ready, missing artifact, red CI and a stale remote head; only a ready predecessor admits its dependent and every failure blocks mutation. *(RM-RELEASE-R005)*

## Quality

Run `pre-commit run --all-files` where feasible, the repository's CCCC staged complexity gate (no new function over cyclomatic 10 or cognitive 15) and the applicable repository quality checks. jscpd and dupehound are not configured repository hooks; record that gap and use a focused duplication review. KISS means one graph builder and one route admission path, with no duplicated release authority.
