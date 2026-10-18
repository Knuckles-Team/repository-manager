"""Regression contracts for the RMDD-04 MCP and CLI adapter extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json

from repository_manager.mcp_server import get_mcp_instance
from repository_manager.mcp_tools import MCP_TOOL_REGISTRY, extend_registry

EXPECTED_REGISTRY = (
    ("git_operations", "GIT_OPERATIONSTOOL"),
    ("misc", "MISCTOOL"),
    ("project_management", "PROJECT_MANAGEMENTTOOL"),
    ("workspace_management", "WORKSPACE_MANAGEMENTTOOL"),
    ("development_surfaces", "DEVELOPMENT_SURFACESTOOL"),
)
EXPECTED_TOOL_NAMES = (
    "repository_manager_add_project",
    "repository_manager_add_projects",
    "repository_manager_build_projects",
    "repository_manager_bulk_bump",
    "repository_manager_bump_version",
    "repository_manager_cleanup_artifacts",
    "repository_manager_clone_projects",
    "repository_manager_clone_repository",
    "repository_manager_commit_code_project",
    "repository_manager_commit_code_projects",
    "repository_manager_commit_project",
    "repository_manager_commit_projects",
    "repository_manager_create_project",
    "repository_manager_discover_projects",
    "repository_manager_generate_markdown_summary",
    "repository_manager_generate_workspace_template",
    "repository_manager_get_consolidated_skill_paths",
    "repository_manager_get_project_map",
    "repository_manager_get_readme",
    "repository_manager_get_workspace_projects",
    "repository_manager_git_action",
    "repository_manager_install_project",
    "repository_manager_install_projects",
    "repository_manager_list_branches",
    "repository_manager_load_projects_from_yaml",
    "repository_manager_maintain_projects",
    "repository_manager_phased_bumpversion",
    "repository_manager_phased_push",
    "repository_manager_pre_commit",
    "repository_manager_pre_commit_projects",
    "repository_manager_pull_project",
    "repository_manager_pull_projects",
    "repository_manager_push_project",
    "repository_manager_push_projects",
    "repository_manager_save_workspace_config",
    "repository_manager_set_threads",
    "repository_manager_setup_from_yaml",
    "repository_manager_test_projects",
    "repository_manager_update_dependency",
    "repository_manager_validate_and_release",
    "repository_manager_validate_single_project",
    "repository_manager_worktree_hygiene",
    "rm_git_clone",
    "rm_git_enumerate",
    "rm_git_pull",
    "rm_git_push",
    "rm_git_phased_push",
    "rm_git_add",
    "rm_git_commit",
    "rm_git_pre_commit",
    "rm_git_commit_code",
    "rm_git_status",
    "rm_git_cancel",
    "rm_git_raw",
    "repository_ingest_repositories",
    "rm_lane_start",
    "rm_lane_doctor",
    "rm_lane_heal",
    "rm_lane_env",
    "rm_lane_finish",
    "rm_worktree_add",
    "rm_worktree_audit",
    "rm_worktree_bulk_add",
    "rm_worktree_list",
    "rm_worktree_merge",
    "rm_worktree_prune",
    "rm_worktree_remove",
    "rm_worktree_reset_branch",
    "rm_worktree_sync",
    "rm_merge_queue_enqueue",
    "rm_merge_queue_status",
    "rm_merge_queue_withdraw",
    "rm_merge_queue_run",
    "rm_merge_queue_config",
    "rm_build_request",
    "rm_build_status",
    "rm_build_artifacts",
    "rm_build_explain",
    "rm_build_gc",
    "rm_projects_install",
    "rm_projects_build",
    "rm_projects_validate",
    "rm_projects_validate_status",
    "rm_docs_readiness_preview",
    "rm_docs_readiness_apply",
    "rm_docs_readiness_verify",
    "rm_gates_run",
    "rm_gates_retest",
    "rm_gates_status",
    "rm_gates_explain",
    "rm_gates_profile",
    "rm_gates_audit_fail_fast",
    "rm_gates_xdist_plan",
    "rm_gates_xdist_apply",
    "rm_gates_phase_direction",
    "rm_gates_repos",
    "rm_gates_phase_direction_here",
    "rm_workspace_list",
    "rm_workspace_list_branches",
    "rm_workspace_setup",
    "rm_workspace_template",
    "rm_workspace_save",
    "rm_workspace_maintain",
    "rm_workspace_maintain_status",
    "rm_workspace_fleet_evidence_check",
    "rm_concepts_reserve",
    "rm_concepts_get",
    "rm_concepts_list",
    "rm_concepts_materialize",
    "rm_concepts_release",
    "rm_concepts_verify_candidate",
    "rm_concepts_verify_generation",
    "rm_concepts_reconcile",
    "rm_remote_workers_register_worker",
    "rm_remote_workers_seed_from_inventory",
    "rm_remote_workers_profile",
    "rm_remote_workers_recheck",
    "rm_remote_workers_stage_source",
    "rm_remote_workers_verify_source",
    "rm_remote_workers_receive_artifact",
    "rm_remote_workers_dispatch_build",
    "rm_remote_workers_host_loss_reconcile",
)

# SHA-256 of the ordered, normalized FastMCP catalog.  This covers names,
# descriptions, tags, parameter schemas, defaults, and required fields
# without duplicating a fixture here.  Originally the RMDD-04 pre-extraction
# baseline; RMDD-20 grew it past that baseline (exposing RMDD-15's
# remote-worker surface and RMDD-17's concept coordination as
# ``rm_remote_workers``/``rm_concepts``); the P0.6/P0.7 invariants program
# grew ``rm_build`` (gained ``host``) and ``rm_remote_workers`` (gained
# ``path`` + expanded ``action`` docs); GOC-60/P0.4 grows it again with
# ``rm_gates`` (the two-tier fast/heavy pre-commit gate driver) registered
# right after ``rm_projects``. NE137 adds ``rm_docs_readiness`` as the
# dry-run-first canonical fleet action. Each growth recomputes the tool name list and
# this digest against the live registry — not drift. Recomputed here for the
# merge of both the P0.6/P0.7 invariants program and GOC-60/P0.4 rm_gates
# landing together; recomputed again for the NFS-buildhost-git-sync lane,
# which wires the already-implemented ``dispatch_build`` action (P0.7's
# ``remote_worker_actions.dispatch_build``, previously reachable only via a
# direct Python import — never through either adapter) into both
# ``rm_remote_workers`` (gained ``command`` + ``workdir``) and the CLI's
# ``--remote-workers`` choices list. Recomputed again for the gate-runner
# lane, which adds ``"retest"`` to ``RM_GATES_ACTIONS`` — the ledger-backed
# narrowed-retest action documented in ``gate_runner.py`` — expanding
# ``rm_gates``'s ``action`` parameter schema/docs. Recomputed once more in the
# same lane for the three gate-CONFIGURATION actions (``audit_fail_fast``,
# ``xdist_plan``, ``xdist_apply``), which route to
# ``fail_fast_audit``/``xdist_rollout`` and add the ``fleet``/``dry_run``
# parameters to the tool signature. Recomputed for RF-ADR-009 §3, which adds
# ``phase_direction`` to ``RM_GATES_ACTIONS`` (routed to
# ``dependency_readiness.check_phase_direction``) and names it in ``rm_gates``'
# ``action`` description; no parameter was added.
# Recomputed again for RF-ADR-009 §3's repository-inferring mode, which adds
# ``phase_direction_here`` to ``RM_GATES_ACTIONS`` (routed to
# ``dependency_readiness.check_phase_direction_here``), names it in
# ``rm_gates``'s ``action`` description, and adds the ``start`` parameter.
# The fleet release gate (RM-RELEASE-R001) adds
# `rm_workspace(action="fleet_evidence_check")` with one bounded
# `fleet_evidence_json` parameter and includes the action in its public schema.
# Recomputed for RM-CONNECTOR-04's certification re-check: `rm_worktree`'s
# `action` parameter gains an explicit `Literal[...]` enumeration of
# `RM_WORKTREE_ACTIONS` (it was a bare `str` before), matching the live
# connector-certify fingerprint for `rm_worktree`
# (`fde09a290e9e0d803d6b1e71de6eeb711da12b6b176f41f0b542ca8ab8f0b0f4`, written to
# `connector_manifest.yml` and `tool_schema_fingerprints.json` by
# `connector-certify --write`, never hand-edited).
# Recomputed for the SDK 0.3.0 per-operation surface: every rm_* action is its own
# rm_<group>_<action> tool, and the Git facade serves one
# repository_manager_<operation> tool per method; no action-routed tool remains.
BASELINE_CATALOG_SHA256 = (
    "1df41dc830731725442940654a64765261b51180a0df7da62713755e820cdeec"
)


def _catalog_payload(tools: list) -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "tags": sorted(tool.tags or []),
        }
        for tool in tools
    ]


def test_registry_order_and_extension_seam_are_deterministic() -> None:
    assert tuple((tag, env) for tag, env, _ in MCP_TOOL_REGISTRY) == EXPECTED_REGISTRY
    assert all(callable(registrar) for _, _, registrar in MCP_TOOL_REGISTRY)

    sentinel = ("future_domain", "FUTURE_DOMAINTOOL", lambda _mcp: None)
    extended = extend_registry(sentinel)
    assert tuple((tag, env) for tag, env, _ in extended) == (
        *EXPECTED_REGISTRY,
        ("future_domain", "FUTURE_DOMAINTOOL"),
    )
    assert len(MCP_TOOL_REGISTRY) == len(EXPECTED_REGISTRY)


def test_cli_entrypoint_remains_a_compatibility_shim() -> None:
    """The packaged CLI entrypoint delegates to the extracted parser module."""
    import repository_manager.repository_manager as repository_manager_module

    assert repository_manager_module.main.__module__ == (
        "repository_manager.repository_manager"
    )
    assert repository_manager_module._run_cli.__module__ == (
        "repository_manager.cli_commands.parser"
    )


def test_per_operation_mcp_catalog_matches_contract() -> None:
    mcp, _args, _middlewares, _registered_tags = get_mcp_instance()
    tools = asyncio.run(mcp.list_tools())
    payload = _catalog_payload(tools)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert tuple(tool["name"] for tool in payload) == EXPECTED_TOOL_NAMES
    assert digest == BASELINE_CATALOG_SHA256
