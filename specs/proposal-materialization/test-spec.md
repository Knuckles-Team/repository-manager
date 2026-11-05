# RM-MATERIALIZE test-spec

Proof level: every rule runs against a temporary Git repository or the `FakeGit` fixture and the fakes in `tests/materialization/fakes.py` (approval authority, receipt store, queue). No token, live graph or queue daemon is required.

## Positive

- `tests/materialization/test_service.py::test_a_passing_gate_commits_with_exactly_the_reviewed_paths`. A valid approval, an allowlisted patch and a passing gate give one commit of exactly the reviewed paths. *(RM-MATERIALIZE-R001, RM-MATERIALIZE-R004)*
- `tests/materialization/test_service.py::test_valid_patch_touches_only_the_approved_path`. A patch to one allowed file changes only that file. *(RM-MATERIALIZE-R003)*
- `tests/materialization/test_service.py::test_queue_acceptance_is_reported_as_queued_never_merged`. The fake queue accepts and the receipt is `queued`. *(RM-MATERIALIZE-R005)*
- `tests/materialization/test_service.py::test_reconcile_promotes_a_landed_candidate_to_merged`. A landed queue record moves the receipt to `merged`. *(RM-MATERIALIZE-R005)*
- `tests/materialization/test_service.py::test_materialization_receipt_contract_round_trips_through_canonical_serialization`. The receipt serializes and re-parses to the same digest. *(RM-MATERIALIZE-R006)*
- `tests/materialization/test_service.py::test_a_replayed_request_returns_the_same_receipt_without_a_second_commit`. A replay after response loss returns one candidate. *(RM-MATERIALIZE-R006)*

## Negative

- `tests/materialization/test_service.py::test_expired_approval_refuses_without_any_worktree`, `::test_approval_bound_to_a_different_proposal_is_unauthorized`, `::test_approval_for_a_different_repository_is_unauthorized`, `::test_approval_from_the_wrong_actor_is_unauthorized` and `::test_approval_authority_rejection_is_unauthorized`: no worktree and no ref. *(RM-MATERIALIZE-R002)*
- `tests/materialization/test_service.py::test_patch_touching_an_unapproved_path_is_rejected`, `::test_patch_header_with_traversal_is_rejected_statically`, `::test_patch_declaring_a_new_symlink_is_rejected`, `::test_binary_patch_content_is_rejected`, `::test_oversized_patch_is_rejected_before_any_git_mutation_can_start`, `::test_stale_base_is_rejected_before_any_worktree_is_created`: the canonical repository is untouched. *(RM-MATERIALIZE-R003)*
- `tests/materialization/test_service.py::test_a_failing_gate_blocks_both_commit_and_queue`. A red gate gives no commit and no queue entry. *(RM-MATERIALIZE-R004)*
- `tests/materialization/test_service.py::test_queue_conflict_is_reported_as_refused_not_success` and `::test_reconcile_reports_a_rejected_candidate_as_refused`. *(RM-MATERIALIZE-R005)*
- `tests/materialization/test_service.py::test_an_altered_replay_under_the_same_key_is_rejected`. *(RM-MATERIALIZE-R006)*
- `tests/materialization/test_service.py::test_unavailable_approval_authority_leaves_the_request_pending`, `::test_unavailable_receipt_store_leaves_the_request_pending` and `::test_unavailable_merge_queue_leaves_a_committed_candidate_pending`. *(RM-MATERIALIZE-R007)*

## Planned

- `tests/test_materialization_cli.py::test_materialize_verb_returns_the_receipt` (planned). The CLI verb reads a proposal and an approval and prints the receipt. *(RM-MATERIALIZE-R001)*
- `tests/test_materialization_mcp.py::test_materialize_tool_returns_the_receipt` (planned). The MCP tool returns the same receipt. *(RM-MATERIALIZE-R001)*
- `tests/materialization/test_service.py::test_validation_profile_selects_the_declared_gate` (planned). The profile named by the proposal selects the gate. *(RM-MATERIALIZE-R004)*
- `tests/materialization/test_receipt_contract.py::test_receipt_carries_gate_result_digest_and_signature` (planned). *(RM-MATERIALIZE-R006)*

## Quality

Run `uv run pytest tests/materialization`, the repository pre-commit, Ruff, mypy and CCCC (no new function over cyclomatic 10 or cognitive 15). One materializer service reuses the existing worktree, gate, commit and queue components, with no alternate Git publisher.
