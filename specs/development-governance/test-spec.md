# RM-GOVERNANCE — Test specification

Proof level: every case runs against disposable Git repositories with deterministic fake clocks and authority adapters; optional forge tests are additive. Rows whose test is marked planned have no test yet.

## Bound tests

- Test: `tests/governance/test_no_parallel_lane_authority.py::test_the_packages_own_lane_module_is_the_live_implementation` and `tests/governance/test_no_parallel_lane_authority.py::test_concept_reservation_module_is_hosted_as_this_packages_own`. *(RM-GOVERNANCE-R001)*
- Test: `tests/governance/test_no_parallel_lane_authority.py::test_no_production_module_imports_the_retired_lane_module`. *(RM-GOVERNANCE-R024)*
- Test: `tests/governance/test_lanes.py::test_gate_still_refuses_a_canonical_commit_in_a_foreign_repo`, `tests/governance/test_lanes.py::test_gate_refuses_a_hand_edited_generated_ledger_view` and `tests/governance/test_lanes.py::test_gate_detects_a_stray_cargo_target_dir_export_in_a_foreign_cargo_repo`. *(RM-GOVERNANCE-R031)*
- Test: `tests/test_windows_portability.py::test_emulated_handles_read_write_and_nest` and the other tests of that module. *(RM-GOVERNANCE-R032)*
- Test: `tests/test_native_lane_authority.py::test_allocate_branch_conflict_refused_by_both`, `tests/test_native_lane_authority.py::test_stale_fence_refused_by_both` and `tests/test_lane_registry.py::test_restart_heartbeat_and_fence_refusal_with_injected_clock`. *(RM-GOVERNANCE-R025)*
- Test: `tests/test_worktree.py::test_concurrent_worktree_creation_keeps_shared_repo_usable_and_not_bare`. *(RM-GOVERNANCE-R026)*
- Test: `tests/test_safe_commit.py::test_safe_commit_stages_only_the_reviewed_paths`, `tests/test_safe_commit.py::test_baseline_binds_the_declared_gate_identity_to_the_committed_tree`, `tests/test_safe_commit.py::test_safe_commit_refuses_without_explicit_paths` and `tests/test_merge_queue.py::test_enqueue_refuses_an_unregistered_checkout`. *(RM-GOVERNANCE-R027)*
- Test: `tests/test_merge_queue.py::test_a_conflict_touching_a_handwritten_file_is_still_rejected`, `tests/test_merge_queue.py::test_a_killed_drain_does_not_strand_the_candidate` and `tests/test_merge_queue.py::test_a_baseline_timeout_is_a_refusal_not_an_empty_baseline`. *(RM-GOVERNANCE-R028)*
- Test: `tests/test_worktree.py::test_audit_unmerged_ahead_is_active`, `tests/test_worktree.py::test_audit_dirty_worktree_is_active`, `tests/test_worktree.py::test_prune_merged_still_removes_a_genuinely_merged_worktree` and `tests/test_worktree.py::test_prune_skips_a_worktree_whose_lane_holds_a_lease`. *(RM-GOVERNANCE-R029)*
- Test: `tests/governance/test_concept_reservation_authority.py::test_fence_owner_and_terminal_visibility_are_enforced`, `tests/governance/test_concept_reservation_authority.py::test_concurrent_clients_have_one_winner_for_one_id` and `tests/governance/test_concept_reservation_authority.py::test_released_id_can_be_reclaimed_but_landed_id_cannot`. *(RM-GOVERNANCE-R030)*
- Test: `tests/test_worktree.py::test_prune_merged_still_removes_a_genuinely_merged_worktree` covers the checkout and branch; `tests/test_worktree.py::test_cleanup_removes_backup_refs_scratch_and_writes_a_log` (planned) covers the rest. *(RM-GOVERNANCE-R003)*
- Test: `tests/test_repository_manager.py::test_managed_checkout_origin_is_sanitized_without_other_config_changes` and `tests/test_repository_manager.py::test_managed_checkout_refuses_failed_origin_sanitization_without_secret_diagnostics`. *(RM-GOVERNANCE-R004)*
- Test: `tests/test_worktree.py::test_worktree_hygiene_prune_removes_only_merged` covers pruning; `tests/test_worktree.py::test_prune_also_prunes_the_package_manager_cache` (planned) covers the cache step. *(RM-GOVERNANCE-R007)*
- Test: `tests/test_worktree.py::test_sync_refuses_when_the_synced_tree_fails_its_declared_gates` and `tests/test_worktree.py::test_sync_reports_a_passing_declared_gate`. *(RM-GOVERNANCE-R013)*
- Test: `tests/test_durable_build_broker.py::test_submission_is_immediate_and_bounded_by_the_same_capacity_profile`. *(RM-GOVERNANCE-R017)*
- Test: `tests/test_dependency_readiness.py::test_find_workspace_manifest_resolves_from_a_location_with_no_ancestor_copy` and `tests/test_dependency_readiness.py::test_dependency_readiness_hook_exits_zero_from_an_isolated_worktree`. *(RM-GOVERNANCE-R018)*
- Test: `tests/test_gates.py::test_one_red_check_refuses_admission_even_though_the_rest_are_green`. *(RM-GOVERNANCE-R019)*
- Test: `tests/test_pre_push_gate.py::test_gate_runs_the_heavy_pre_push_stage`. *(RM-GOVERNANCE-R022)*
- Test: `tests/governance/test_lanes.py::test_expired_lease_is_reclaimed` and `tests/test_resource_lease.py::test_expired_fence_releases_slot` (planned). *(RM-GOVERNANCE-R023.5)*

## Planned

- Test: `tests/test_governance_hooks.py::test_readme_deploy_block_runs_from_a_linked_worktree` (planned). *(RM-GOVERNANCE-R002)*
- Test: `tests/test_check_complexity_staged.py::test_match_dispatch_function_is_scored_by_its_structure` (planned). *(RM-GOVERNANCE-R005)*
- Test: `tests/test_governance_hooks.py::test_duplication_gate_reports_zero_new_pairs_on_the_pushed_tree` (planned). *(RM-GOVERNANCE-R006)*
- Test: `tests/test_governance_hooks.py::test_kiss_hook_resolves_a_test_mod_declaration` (planned). *(RM-GOVERNANCE-R008)*
- Test: `tests/test_governance_hooks.py::test_comment_only_commit_is_refused_by_the_duplication_gate` (planned). *(RM-GOVERNANCE-R009)*
- Test: `tests/test_governance_hooks.py::test_only_registered_pairs_are_exempt` (planned). *(RM-GOVERNANCE-R010)*
- Test: `tests/test_build_status_page.py::test_builder_leaves_no_diff_against_the_committed_page` (planned). *(RM-GOVERNANCE-R011)*
- Test: `tests/test_remote_execution_source_staging.py::test_content_hash_detects_a_change_where_git_diff_reports_none` (planned). *(RM-GOVERNANCE-R012)*
- Test: `tests/test_gates.py::test_full_hook_output_is_captured_and_an_interrupted_run_is_not_a_pass` (planned). *(RM-GOVERNANCE-R014)*
- Test: `tests/test_governance_protocol.py::test_protocol_names_review_merge_notify_rebase_in_order` (planned). *(RM-GOVERNANCE-R015)*
- Test: `tests/test_worktree.py::test_remove_deletes_backup_refs_and_build_scratch_of_the_branch` (planned). *(RM-GOVERNANCE-R016)*
- Test: `tests/test_governance_hooks.py::test_split_file_reports_zero_new_pairs_against_the_base` (planned). *(RM-GOVERNANCE-R020)*
- Test: `tests/test_governance_hooks.py::test_secret_exception_self_check_matches_only_the_pinned_occurrence` (planned). *(RM-GOVERNANCE-R021)*
- Test: `tests/test_resource_lease.py::test_host_lease_fields` (planned). *(RM-GOVERNANCE-R023.1)*
- Test: `tests/test_resource_lease.py::test_capacity_read_from_inventory` (planned). *(RM-GOVERNANCE-R023.2)*
- Test: `tests/test_resource_lease.py::test_slot_counted_per_host` (planned). *(RM-GOVERNANCE-R023.3)*
- Test: `tests/test_resource_lease.py::test_all_hosts_full_refuses` (planned). *(RM-GOVERNANCE-R023.4)*
- Test: `tests/test_resource_lease.py::test_gate_runner_acquires_and_releases` (planned). *(RM-GOVERNANCE-R023.6)*

## Quality

Run `pre-commit run --all-files`, the applicable CCCC staged complexity check (no new function over cyclomatic 10 or cognitive 15), Ruff, mypy and pytest, and a duplication review. jscpd and dupehound are not configured by this repository's ordinary hooks; record their availability instead of asserting a pass. KISS means one lane registry, one queue and one concept claim authority, with adapters rather than duplicate implementations. Save the exact commit and CI receipts and an import inventory when accepting.
