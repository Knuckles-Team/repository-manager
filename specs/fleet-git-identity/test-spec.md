# RM-IDENTITY test-spec

Proof level: every case runs against temporary Git repositories built by `tests/history_identity_git_fixtures.py` (synthetic users, branches, annotated tags and a local bare remote). No credential and no fleet checkout is required. The CLI tests drive the real parser and dispatch of `repository_manager.governance.cli.main`.

## Positive

- `tests/test_history_identity_discovery.py::test_discover_enumerates_branches_tags_remotes_and_notes` and `::test_discover_is_deterministic_for_unchanged_state`. *(RM-IDENTITY-R003)*
- `tests/test_history_identity_preview.py::test_generate_preview_is_deterministic_across_runs` and `tests/test_history_identity_cli.py::test_history_identity_cli_output_is_deterministic_across_two_runs`. *(RM-IDENTITY-R004)*
- `tests/test_history_identity_approval.py::test_verify_approval_accepts_a_valid_matching_signature`. *(RM-IDENTITY-R005)*
- `tests/test_history_identity_consistency.py::test_check_history_consistency_passes_for_a_normal_branch`. *(RM-IDENTITY-R002)*

## Negative

- `tests/test_history_identity_discovery.py::test_discover_refuses_hidden_ref`, `::test_discover_refuses_unreadable_ref`, `::test_discover_refuses_unreachable_remote` and `::test_discover_refuses_shallow_clone`. *(RM-IDENTITY-R003)*
- `tests/test_history_identity_consistency.py::test_check_history_consistency_detects_an_orphan_branch` and `::test_admit_for_rewrite_plan_raises_on_a_diverged_report`. *(RM-IDENTITY-R002)*
- `tests/test_history_identity_preview.py::test_generate_preview_digest_changes_when_source_mutates` and `::test_generate_preview_surfaces_collision_warning_for_unmapped_identity`. *(RM-IDENTITY-R004)*
- `tests/test_history_identity_approval.py::test_verify_approval_rejects_missing_approval`, `::test_verify_approval_rejects_expired_approval`, `::test_verify_approval_rejects_a_forged_signature` and `::test_verify_approval_rejects_the_wrong_key`; `tests/test_history_identity_cli.py::test_history_identity_cli_refuses_an_approval_bound_to_a_different_plan`. *(RM-IDENTITY-R005)*
- `tests/test_history_identity_cli.py::test_history_identity_cli_mutates_nothing`. The preview leaves refs, HEAD and the object count unchanged. *(RM-IDENTITY-R004)*

## Planned

- `tests/test_history_identity_rewrite.py::test_rewrite_preserves_trees_and_backup_refs_restore_history` (planned). *(RM-IDENTITY-R001)*
- `tests/test_history_identity_rewrite.py::test_missing_expired_or_altered_approval_mutates_nothing` (planned). *(RM-IDENTITY-R005)*
- `tests/test_history_identity_verify.py::test_every_rewritten_commit_keeps_tree_parents_message_and_timestamps` (planned): merges, author and committer aliases, annotated tags, signed objects refused until re-signing is authorized. *(RM-IDENTITY-R006)*
- `tests/test_history_identity_transaction.py::test_failure_before_update_leaves_old_refs_and_backups_restore_after_update` (planned): a failure at every phase leaves the old refs or a complete rollback anchor. *(RM-IDENTITY-R007)*
- `tests/test_history_identity_publish.py::test_remote_drift_and_rejected_branch_yield_partial_receipt_not_success` (planned): two local bare remotes, one drifts or rejects; a retry does not overwrite the drift. *(RM-IDENTITY-R008)*
- `tests/test_history_identity_coordination.py::test_no_queued_candidate_merges_against_stale_history` (planned). *(RM-IDENTITY-R009)*

## Quality

Run `uv run pytest tests -k history_identity`, the repository pre-commit, Ruff, mypy and CCCC (no new function over cyclomatic 10 or cognitive 15). The rewrite verifier is a separate module from the rewrite engine, so the engine cannot grade its own output. One policy evaluator (`IdentityPolicy`) and one rewrite transaction service exist, with no per-remote duplicated logic. jscpd and dupehound are not configured repository hooks; run them where available and report a missing tool.
