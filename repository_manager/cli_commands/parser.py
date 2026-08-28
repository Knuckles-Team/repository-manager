"""Repository Manager CLI parser and command routing.

This module owns argument construction and dispatch marshalling. It deliberately
accepts a CliRuntime so the packaged entrypoint and tests retain the same Git
factory and configuration seams without importing parser policy into the Git core.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from repository_manager.cli_commands.build_queue import run_build_queue_cli
from repository_manager.cli_commands.concepts import run_concepts_cli
from repository_manager.cli_commands.context import CliRuntime
from repository_manager.cli_commands.differential_selection import (
    run_differential_select_cli,
)
from repository_manager.cli_commands.docs_readiness import run_docs_readiness_cli
from repository_manager.cli_commands.lane import run_lane_cli
from repository_manager.cli_commands.merge_queue import run_merge_queue_cli
from repository_manager.cli_commands.remote_workers import run_remote_workers_cli


def run(runtime: CliRuntime) -> None:
    """
    Main entry point for the Repository Manager CLI.
    Supports workspace management, Git bulk operations, and maintenance.
    """
    parser = argparse.ArgumentParser(
        description="Repository Manager - 100% Model-Driven Pydantic Graph Agent",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Standard setup (clones all missing repos in parallel)
  repository-manager --clone

  # Maintenance workflow (Bump patch version everywhere)
  repository-manager --maintain --bump patch

  # Selective operations
  repository-manager --repositories "genius-agent, gitlab-api" --pull
  """,
    )

    group_general = parser.add_argument_group("General Options")
    group_general.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {runtime.version}"
    )
    group_general.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to workspace.yml file (Standard Source).",
        default=runtime.default_workspace_yml,
    )
    group_general.add_argument(
        "-w",
        "--workspace",
        type=str,
        help="Path to the workspace root directory (default: ~/Workspace).",
        default=runtime.default_workspace,
    )
    group_general.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Parallel thread count (default: 6).",
        default=runtime.default_threads,
    )
    group_general.add_argument(
        "-r",
        "--repositories",
        type=str,
        help="Comma-separated list of repository names to filter operations.",
    )

    group_workspace = parser.add_argument_group("Workspace Management")
    group_workspace.add_argument(
        "--setup",
        action="store_true",
        help="Initialize workspace: directory structure & clones missing repos.",
    )

    group_workspace.add_argument(
        "--save",
        action="store_true",
        help="Save current in-memory config back to YAML (Updates).",
    )

    group_workspace.add_argument(
        "--branches",
        action="store_true",
        help="List the active git branch for all projects.",
    )
    manifest_gate = group_workspace.add_mutually_exclusive_group()
    manifest_gate.add_argument(
        "--manifest-check",
        action="store_true",
        help="Validate the canonical manifest and fail when either mirror has drifted.",
    )
    manifest_gate.add_argument(
        "--manifest-sync",
        action="store_true",
        help="Mirror the canonical manifest to the runtime and packaged seed paths.",
    )
    group_workspace.add_argument(
        "--manifest-source",
        type=str,
        help="Explicit canonical root workspace.yml; packaged seeds are never sources.",
    )
    group_workspace.add_argument(
        "--manifest-runtime-destination",
        type=str,
        help="Override the Graph-OS runtime manifest destination.",
    )
    group_workspace.add_argument(
        "--manifest-seed-destination",
        type=str,
        help="Override the packaged repository-manager seed destination.",
    )
    group_workspace.add_argument(
        "--manifest-dry-run",
        action="store_true",
        help="Report manifest synchronization changes without writing either destination.",
    )
    group_workspace.add_argument(
        "--manifest-profile",
        type=str,
        help="Show the repositories selected by a named bootstrap profile.",
    )
    group_workspace.add_argument(
        "--manifest-selector",
        action="append",
        default=[],
        help="Add a named bootstrap selector (may be repeated).",
    )

    group_git = parser.add_argument_group("Git Bulk Operations (Parallelized)")
    group_git.add_argument(
        "--clone",
        action="store_true",
        help="Clone all missing repositories in the workspace.",
    )
    group_git.add_argument(
        "--pull", action="store_true", help="Pull latest changes for all projects."
    )
    group_git.add_argument(
        "--add",
        action="store_true",
        help="Stage all changes in the specified repositories.",
    )
    group_git.add_argument(
        "--commit",
        action="store_true",
        help="Commit staged changes in the specified repositories.",
    )
    group_git.add_argument(
        "-m",
        "--message",
        type=str,
        help="Commit message for bulk commits. Required for --commit.",
    )
    group_git.add_argument(
        "--default-branch",
        action="store_true",
        help="Switch all repos to their default branch (via origin/HEAD).",
    )

    # CONCEPT:RM-LANE-DOCTOR — the lane lifecycle, executable.
    # A deliberate sibling of the worktree verbs and the merge queue: a lane
    # STARTS here (isolated worktree + partitioned environment), checks itself
    # here while working, and FINISHES here by handing its branch to the queue.
    group_lane = parser.add_argument_group(
        "Lane Lifecycle (isolation preflight for concurrent agents and humans)"
    )
    group_lane.add_argument(
        "--lane",
        choices=["doctor", "start", "finish", "env", "heal"],
        help=(
            "'doctor' checks this tree's isolation and mutates nothing; 'start' "
            "opens an isolated worktree and proves its partitions; 'env' prints "
            "the shell exports; 'finish' preflights then enqueues the branch for "
            "landing; 'heal' diagnoses like 'doctor' then REPAIRS the core.bare/"
            "index-collapse class of finding itself (own tree + canonical), "
            "rather than only naming a remedy to run by hand. Run 'doctor' "
            "whenever something behaves impossibly; reach for 'heal' once it has."
        ),
    )
    group_lane.add_argument(
        "--lane-path",
        type=str,
        default=None,
        help="The lane's working tree for --lane doctor/env/finish (default: cwd).",
    )
    group_lane.add_argument(
        "--lane-repo",
        type=str,
        default="",
        help="Repository basename or path for --lane start.",
    )
    group_lane.add_argument(
        "--lane-branch",
        type=str,
        default="",
        help="Branch for --lane start/finish (finish defaults to the tree's HEAD).",
    )
    group_lane.add_argument(
        "--lane-base",
        type=str,
        default="",
        help="Base branch to fork from / land onto (default: main).",
    )
    group_lane.add_argument(
        "--lane-shell",
        action="store_true",
        help='Print `export K=V` lines instead of JSON, for `eval "$(...)"`.',
    )
    group_lane.add_argument(
        "--lane-force",
        action="store_true",
        help="For --lane finish: enqueue despite a blocking preflight check.",
    )

    group_maintenance = parser.add_argument_group("Maintenance Lifecycle")
    group_maintenance.add_argument(
        "--install",
        action="store_true",
        help="Run 'pip install --break-system-packages -e .' for all projects.",
    )
    group_maintenance.add_argument(
        "--build", action="store_true", help="Run 'python -m build' for all projects."
    )
    group_maintenance.add_argument(
        "--validate",
        action="store_true",
        help="Run comprehensive pre-release validation.",
    )
    group_maintenance.add_argument(
        "--no-report",
        action="store_true",
        help="Do not automatically generate the validation report directory.",
    )
    group_maintenance.add_argument(
        "--type",
        choices=[
            "all",
            "static-analysis",
            "runtime-validation",
            "mcp",
            "agent",
            "flat",
            "graph",
            "test",
            "pre-commit",
        ],
        default="all",
        help="Filter validation mode or target (default: all).",
    )
    group_maintenance.add_argument(
        "--pre-commit",
        action="store_true",
        help="Run pre-commit checks and autoupdate hooks.",
    )
    group_maintenance.add_argument(
        "--gate",
        choices=["fast", "heavy"],
        default=None,
        help=(
            "Run the two-tier pre-commit gate across targeted projects and print "
            "a summary: 'fast' -> `pre-commit run --hook-stage pre-commit` "
            "(formatters/linters); 'heavy' -> `--hook-stage pre-push` (pytest, "
            "cargo, `uv lock --check`, ...). The CLI counterpart of the MCP "
            "`rm_gates` tool's 'run' action."
        ),
    )
    group_maintenance.add_argument(
        "--gate-retest",
        choices=["fast", "heavy"],
        default=None,
        help=(
            "Like --gate, but narrows the re-run to whatever "
            "repository_manager.gate_ledger last recorded as FAILING for each "
            "targeted repo/stage instead of re-running the whole wave -- the "
            "CLI counterpart of the MCP `rm_gates` tool's 'retest' action "
            "(measured incident 2026-08-21: re-running a full 90-minute heavy "
            "wave to validate each of six failing hooks cost one push ~6 hours). "
            "A repo with no prior recorded run still gets the FULL wave (never "
            "silently treated as clean); a stale baseline (recorded against a "
            "different commit than HEAD) is never trusted either and also "
            "degrades to the full wave. On an all-pass narrowed retest, a "
            "second full-wave job is submitted automatically -- a narrowed "
            "pass alone is never sufficient evidence of shippability (see "
            "gate_ledger.GateLedger.is_shippable). Pass --same-node to assert "
            "this invocation is colocated with whatever produced the ledger's "
            "last recorded run; it is carried onto each submitted job for a "
            "future record_run writer to honor -- this command's own ledger "
            "reads are already local and do not depend on it."
        ),
    )
    group_maintenance.add_argument(
        "--maintain",
        action="store_true",
        help="Execute phased maintenance (Bump -> Pre-commit -> Verify).",
    )
    group_maintenance.add_argument(
        "--push",
        action="store_true",
        help="Execute phased push.",
    )
    group_maintenance.add_argument(
        "--bump",
        choices=["patch", "minor", "major"],
        help="Version bump part (major/minor/patch). Use with --maintain or standalone.",
    )
    group_maintenance.add_argument(
        "--report",
        nargs="?",
        const=True,
        help="Export markdown summary results to a file (default: Workspace root).",
    )
    group_maintenance.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Starting phase for maintenance lifecycle (1-3).",
    )
    group_maintenance.add_argument(
        "--single-phase",
        action="store_true",
        help="Only execute the specified phase, do not proceed to subsequent phases.",
    )
    group_maintenance.add_argument(
        "--no-auto-start",
        dest="auto_start",
        action="store_false",
        default=True,
        help=(
            "Opt out of change-aware start. By default --maintain/--push begin at "
            "the lowest phase with repository changes; this forces a start at "
            "--phase instead."
        ),
    )
    group_maintenance.add_argument(
        "--project",
        type=str,
        help="Only execute maintenance operations for a specific project.",
    )
    group_maintenance.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform maintenance operations without committing changes. Use with --maintain.",
    )
    group_maintenance.add_argument(
        "--allow-pre-commit",
        action="store_true",
        help="Execute the pre-commit phase during maintenance workflow. Pre-commits are skipped by default.",
    )
    group_maintenance.add_argument(
        "--config",
        type=str,
        help="Path to an overridden maintenance JSON/YAML configuration. Use with --maintain.",
    )
    group_maintenance.add_argument(
        "--break-system-packages",
        action="store_true",
        help="Include --break-system-packages in pip install commands.",
    )

    # CONCEPT:RM-MERGE-QUEUE — the cross-project parallel-development driver.
    # Deliberately a sibling of the worktree verbs: a lane takes a worktree, works,
    # then hands the branch to the queue, and the queue lands it and prunes both.
    group_queue = parser.add_argument_group(
        "Merge Queue (per-repo gates declared in .mergequeue.yaml)"
    )
    group_queue.add_argument(
        "--merge-queue",
        choices=["enqueue", "status", "withdraw", "run", "config"],
        help=(
            "Serialized merge queue for ANY repository. Gates come from that "
            "repo's own .mergequeue.yaml, never from this tool. Use --repo-path "
            "to select the repository (default: cwd). Exit 75 means another "
            "runner holds the lease -- defer, do not proceed."
        ),
    )
    group_queue.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="A working tree of the target repository for --merge-queue (default: cwd).",
    )
    group_queue.add_argument(
        "--queue-branch",
        type=str,
        default="",
        help="Candidate branch for --merge-queue enqueue/withdraw (default: current HEAD).",
    )
    group_queue.add_argument(
        "--queue-base",
        type=str,
        default="",
        help="Branch to land onto (default: the repo's declared base).",
    )
    group_queue.add_argument(
        "--queue-reason",
        type=str,
        default="",
        help="Why a candidate is being withdrawn.",
    )
    group_queue.add_argument(
        "--queue-batch-size",
        type=int,
        default=0,
        help="Candidates gated together per run (0 = the repo's declared batch_size).",
    )
    group_queue.add_argument(
        "--queue-no-prune",
        action="store_true",
        help="Land but keep the worktrees and branches (skips the guarded prune).",
    )

    # CONCEPT:RM-DIFF-SELECT — the GOC-69 pre-push differential tier's mapping
    # step (changed files -> pytest targets). A thin standalone way to inspect
    # a selection; the wired consumer is rm_gates(action=run, stage=heavy).
    # Reuses --repo-path from the queue group above rather than declaring its own.
    group_diff = parser.add_argument_group(
        "Differential Test Selection (GOC-69 pre-push tier)"
    )
    group_diff.add_argument(
        "--differential-select",
        action="store_true",
        help=(
            "Print, as JSON, the pytest targets a differential pre-push run "
            "would select for the diff between --diff-base and --diff-ref "
            "(merge-base relative, same computation the merge queue uses). "
            "Fails OPEN to the full suite whenever narrowing is not provably "
            "safe -- see repository_manager/differential_selection.py."
        ),
    )
    group_diff.add_argument(
        "--diff-base", type=str, default="main", help="Base ref (default: main)."
    )
    group_diff.add_argument(
        "--diff-ref", type=str, default="HEAD", help="Candidate ref (default: HEAD)."
    )
    group_diff.add_argument(
        "--diff-src-roots",
        type=str,
        default=".",
        help="Comma-separated import roots (default: '.').",
    )
    group_diff.add_argument(
        "--diff-test-roots",
        type=str,
        default="tests",
        help="Comma-separated test roots (default: 'tests').",
    )
    group_diff.add_argument(
        "--diff-fanin-threshold",
        type=int,
        default=25,
        help="Reverse-import fan-in above which a module is treated as a hub (default: 25).",
    )

    # CONCEPT:RM-TASK-LEDGER — the content-addressed build broker. A sibling of
    # --merge-queue: same "declared per repo, refused when absent" contract,
    # same dispatch()-core-shared-by-every-surface shape, different resource
    # (a build's output artifacts, not a landed branch).
    group_build = parser.add_argument_group(
        "Build Broker (per-repo build specs declared in .buildcache.yaml)"
    )
    group_build.add_argument(
        "--build-broker",
        choices=["request", "status", "artifacts", "explain", "gc"],
        help=(
            "Content-addressed build broker for ANY repository (distinct from "
            "the packaging '--build' above). A second request for the SAME "
            "(repo, tree-sha, feature-set, toolchain, target) waits on and "
            "reuses the first's published artifacts instead of rebuilding. "
            "Use --repo-path to select the repository (default: cwd)."
        ),
    )
    group_build.add_argument(
        "--build-spec",
        type=str,
        default="",
        help="Which declared spec to build (default: the repo's first).",
    )
    group_build.add_argument(
        "--build-key",
        type=str,
        default="",
        help="A cache key, for --build-broker status/artifacts/explain.",
    )
    group_build.add_argument(
        "--same-node",
        action="store_true",
        help=(
            "Assert this invocation runs on the SAME node as the target repo's "
            "lease holder -- only pass this when it is actually true (this IS "
            "the pinned repository-manager-mcp process, or an operator has "
            "verified pinning). An unproven assertion reintroduces the exact "
            "false-safety this flag exists to prevent; unset, the broker "
            "refuses and names the MCP route instead."
        ),
    )
    group_build.add_argument(
        "--build-wait-timeout",
        type=int,
        default=60,
        help="Seconds to wait on an in-flight build of the same key before building anyway.",
    )
    group_build.add_argument(
        "--build-keep-recent",
        type=int,
        default=10,
        help="For --build-broker gc: always keep this many most-recent cache entries.",
    )
    group_build.add_argument(
        "--build-max-age-days",
        type=int,
        default=14,
        help="For --build-broker gc: reclaim cache entries older than this (subject to --build-keep-recent).",
    )
    group_build.add_argument(
        "--build-host",
        type=str,
        default=None,
        help=(
            "For --build-broker request: dispatch to this REGISTERED, "
            "authorized remote host instead of building locally. Requires a "
            "clean local tree; does not yet retrieve artifacts back. Omit "
            "to build locally (the default)."
        ),
    )

    # RMDD-20 — exposing RMDD-17's concept-claim coordination. The CLI and
    # the MCP `rm_concepts` tool call the exact same
    # `repository_manager.concept_actions.dispatch` core (parity chokepoint).
    group_concepts = parser.add_argument_group(
        "Concept-Claim Coordination (RMDD-17 via RMDD-20)"
    )
    group_concepts.add_argument(
        "--concepts",
        choices=[
            "reserve",
            "list",
            "get",
            "release",
            "materialize",
            "verify_candidate",
            "verify_generation",
            "reconcile",
        ],
        help="Concept-id claim lifecycle action against RMDD-16's injected authority.",
    )
    group_concepts.add_argument(
        "--concepts-repo-root",
        type=str,
        default=".",
        help="Repository working tree root for --concepts (default: cwd).",
    )
    group_concepts.add_argument(
        "--concepts-tenant-ref",
        type=str,
        default="",
        help="Authenticated tenant scope for --concepts.",
    )
    group_concepts.add_argument(
        "--concepts-lane-ref",
        type=str,
        default="",
        help="Lane/worktree identity for --concepts fragment provenance.",
    )
    group_concepts.add_argument(
        "--concepts-params-json",
        type=str,
        default="",
        help=(
            "JSON object with the action's remaining fields (concept_id, "
            "namespace, owner_ref, reservation_id, candidate, ...) -- the "
            "same fields the rm_concepts MCP tool accepts."
        ),
    )

    # RMDD-20 — exposing RMDD-15's remote worker registry/staging/artifact
    # transport. The CLI and the MCP `rm_remote_workers` tool call the exact
    # same `repository_manager.remote_worker_actions.dispatch` core.
    group_remote_workers = parser.add_argument_group(
        "Remote Worker Registry, Staging, and Artifacts (RMDD-15 via RMDD-20)"
    )
    group_remote_workers.add_argument(
        "--remote-workers",
        choices=[
            "register_worker",
            "seed_from_inventory",
            "profile",
            "recheck",
            "stage_source",
            "verify_source",
            "receive_artifact",
            "host_loss_reconcile",
            "dispatch_build",
        ],
        help="Remote-worker registry/source-staging/artifact-transport action.",
    )
    group_remote_workers.add_argument(
        "--remote-workers-params-json",
        type=str,
        default="",
        help=(
            "JSON object with the action's fields (host_id, origin, "
            "tree_sha, relative_path, content_base64, ...) -- the same "
            "fields the rm_remote_workers MCP tool accepts."
        ),
    )

    group_docs_readiness = parser.add_argument_group(
        "Documentation Readiness Fleet Action"
    )
    group_docs_readiness.add_argument(
        "--docs-readiness",
        nargs="?",
        const="preview",
        choices=["preview", "apply", "verify"],
        help=(
            "Preview (default), apply, or verify canonical agent-readiness artifacts; "
            "readiness config must already be generated/adopted per repository."
        ),
    )
    group_docs_readiness.add_argument(
        "--docs-readiness-repository",
        type=str,
        default=None,
        help="Exact agent-packages workspace.yml identity; required for apply.",
    )
    group_docs_readiness.add_argument(
        "--docs-readiness-confirm",
        action="store_true",
        help="Confirm the exact repository apply requested by --docs-readiness apply.",
    )

    args = parser.parse_args()

    # Handled before every other verb and returns immediately: the queue drives a
    # SINGLE named repository and must not be combined with the workspace-wide
    # bulk operations above, whose --path/--repositories semantics are different.
    _dispatch_immediate_verb(args)

    if _dispatch_manifest_ops(runtime, parser, args):
        return

    # CONCEPT:RM-LANE-DOCTOR — handled before the bulk verbs and returning
    # immediately: the lane verbs drive ONE tree and must not be combined with
    # the workspace-wide operations below, whose --path/--repositories semantics
    # are different.
    if args.lane:
        sys.exit(run_lane_cli(args))

    git = runtime.git_factory(
        path=args.workspace if args.workspace != runtime.default_workspace else None,
        threads=args.threads,
        report_path=args.report,
    )
    _apply_runtime_flags(git, args)
    _load_workspace_file(runtime, parser, git, args)
    _apply_repository_filter(git, args)
    _maybe_setup_from_file(runtime, git, args)
    _run_clone_pull(git, args)

    _run_basic_bulk_verbs(runtime, git, args)

    has_errors = _run_gate_dispatch(runtime, git, args)
    has_errors = _dispatch_validate(runtime, git, args, has_errors)
    has_errors = _dispatch_bump(runtime, git, args, has_errors)
    has_errors = _dispatch_maintain(runtime, git, args, has_errors)
    _dispatch_push(runtime, git, args, has_errors)


def _dispatch_immediate_verb(args: argparse.Namespace) -> None:
    """Verbs handled before every other option; each exits the process.

    Extracted verbatim (same order, same conditions) from the head of
    ``run``'s dispatch logic, immediately after ``parser.parse_args()``.
    ``sys.exit`` raises ``SystemExit``, which propagates out of this call
    exactly as it did when inlined directly in ``run``.
    """

    if args.merge_queue:
        sys.exit(run_merge_queue_cli(args))
    if args.differential_select:
        sys.exit(run_differential_select_cli(args))
    if args.build_broker:
        sys.exit(run_build_queue_cli(args))
    if args.concepts:
        sys.exit(run_concepts_cli(args))
    if args.remote_workers:
        sys.exit(run_remote_workers_cli(args))
    if args.docs_readiness:
        sys.exit(run_docs_readiness_cli(args))


def _dispatch_manifest_ops(
    runtime: CliRuntime, parser: argparse.ArgumentParser, args: argparse.Namespace
) -> bool:
    """Validate manifest flags and run the manifest gate.

    Extracted verbatim from ``run``. Returns ``True`` only where the
    original inline code executed a bare ``return`` -- ``run`` must return
    immediately in that case, exactly as before.
    """

    manifest_option_used = any(
        (
            args.manifest_source,
            args.manifest_runtime_destination,
            args.manifest_seed_destination,
            args.manifest_dry_run,
            args.manifest_profile,
            args.manifest_selector,
        )
    )
    if manifest_option_used and not (args.manifest_check or args.manifest_sync):
        parser.error(
            "manifest source, destinations, profiles, selectors, and dry-run "
            "require --manifest-check or --manifest-sync"
        )
    if args.manifest_dry_run and not args.manifest_sync:
        parser.error("--manifest-dry-run requires --manifest-sync")

    if args.manifest_check or args.manifest_sync:
        return _run_manifest_sync(runtime, parser, args)
    return False


def _run_manifest_sync(
    runtime: CliRuntime, parser: argparse.ArgumentParser, args: argparse.Namespace
) -> bool:
    if not args.manifest_source:
        parser.error("--manifest-source is required for the manifest gate")
    try:
        report = runtime.synchronize_workspace_manifest(
            args.manifest_source,
            runtime_destination=args.manifest_runtime_destination,
            seed_destination=args.manifest_seed_destination,
            check=args.manifest_check,
            dry_run=args.manifest_dry_run,
            profile=args.manifest_profile,
            selectors=args.manifest_selector,
        )
    except runtime.manifest_error as exc:
        parser.error(str(exc))
    print(json.dumps(report.as_dict(), sort_keys=True))
    if args.manifest_check and not report.synchronized:
        raise SystemExit(1)
    return True


def _apply_runtime_flags(git: object, args: argparse.Namespace) -> None:
    if args.default_branch:
        git.set_to_default_branch = True

    if args.threads:
        git.set_threads(threads=args.threads)


def _load_workspace_file(
    runtime: CliRuntime,
    parser: argparse.ArgumentParser,
    git: object,
    args: argparse.Namespace,
) -> None:
    if args.file:
        if os.path.exists(args.file):
            if not git.load_projects_from_yaml(args.file):
                runtime.logger.warning("Could not load the requested Workspace YAML")
        else:
            runtime.logger.error("Workspace configuration file was not found")
            parser.print_help()
            sys.exit(2)

    if not git.project_map and os.path.exists(runtime.default_workspace_yml):
        git.load_projects_from_yaml(runtime.default_workspace_yml)


def _filter_existing_project_map(git: object, names_to_keep: set[str]) -> None:
    filtered = {}
    for url, path in git.project_map.items():
        name = url.split("/")[-1].replace(".git", "")
        if name in names_to_keep:
            filtered[url] = path
    git.project_map = filtered


def _seed_project_map_from_names(git: object, repositories: list[str]) -> None:
    for r in repositories:
        if "/" in r:
            name = r.split("/")[-1].replace(".git", "")
            git.project_map[r] = os.path.join(git.path, name)
        else:
            git.project_map[os.path.join("https://github.com/", r)] = os.path.join(
                git.path, r
            )


def _apply_repository_filter(git: object, args: argparse.Namespace) -> None:
    if not args.repositories:
        return
    repositories = args.repositories.replace(" ", "").split(",")
    names_to_keep = set(repositories)
    if git.project_map:
        _filter_existing_project_map(git, names_to_keep)
    else:
        _seed_project_map_from_names(git, repositories)


def _maybe_setup_from_file(
    runtime: CliRuntime, git: object, args: argparse.Namespace
) -> None:
    if args.file and os.path.exists(args.file):
        if args.setup:
            runtime.logger.info("Setting up workspace from configured manifest")
            git.load_projects_from_yaml(args.file)


def _run_clone_pull(git: object, args: argparse.Namespace) -> None:
    if args.clone:
        git.clone_projects()
    if args.pull:
        git.pull_projects()


def _run_basic_bulk_verbs(
    runtime: CliRuntime, git: object, args: argparse.Namespace
) -> None:
    if args.add:
        results = git.add_projects()
        summary = git.generate_markdown_summary("Bulk Git Add", results)
        runtime.logger.info(summary)
        git._export_report(summary, "git_add_report.md")

    if args.commit:
        if not args.message:
            runtime.logger.error(
                "Error: --message/-m is required for bulk commits when using --commit."
            )
            sys.exit(1)
        results = git.commit_projects(message=args.message)
        summary = git.generate_markdown_summary("Bulk Git Commit", results)
        runtime.logger.info(summary)
        git._export_report(summary, "git_commit_report.md")

    if args.branches:
        branches = git.list_branches()
        runtime.logger.info("\n--- Workspace Branches ---")
        for _proj, _branch in sorted(branches.items()):
            runtime.logger.info("Configured project branch discovered")

    if args.pre_commit:
        git.pre_commit_projects(run=True, autoupdate=True)

    if args.install:
        results = git.install_projects()
        summary = git.generate_markdown_summary("Installation", results)
        runtime.logger.info(summary)
        git._export_report(summary, "install_report.md")

    if args.build:
        results = git.build_projects()
        summary = git.generate_markdown_summary("Build", results)
        runtime.logger.info(summary)
        git._export_report(summary, "build_report.md")


def _run_gate_dispatch(
    runtime: CliRuntime, git: object, args: argparse.Namespace
) -> bool:
    """Run --gate / --gate-retest. Extracted verbatim from ``run``, including
    its three nested closures (unchanged) -- only the surrounding statements
    moved; lizard measures nested ``def``s as their own units, so moving them
    does not change what was already being measured separately from ``run``.
    """

    has_errors = False

    if args.gate or args.gate_retest:
        from repository_manager import gate_runner
        from repository_manager.gate_ledger import default_gate_ledger
        from repository_manager.gates import explain_gate_result, run_gate_stage
        from repository_manager.scan_models import RepoScanResult

        # One LocalJobStore per invocation: gate_runner.dispatch's `run`/
        # `retest` need SOMETHING that satisfies the submit_job/jobs/jobs_lock
        # contract the MCP job store provides, and the CLI has no background
        # server to reuse. LocalJobStore runs each job synchronously inline
        # (see its docstring); _fan_out's own bounded pool is what makes
        # this still run repos in parallel -- the exact ThreadPoolExecutor
        # that used to be duplicated here directly, now shared with the MCP
        # adapter via gate_runner._fan_out.
        job_store = gate_runner.LocalJobStore()

        def _resolve_targets(
            threads_: int | None, repos_: str | None
        ) -> list[tuple[str, str]]:
            return gate_runner.targets_from_project_map(git.project_map, repos_)

        def _make_submit_one(stage: str):
            def submit_one(
                repo_name: str,
                path: str,
                *,
                hook_ids: list[str] | None = None,
                trigger: str = "run",
                scope: str = "full_wave",
                _escalate_on_pass: bool = False,
                _same_node: bool = False,
            ) -> dict[str, str]:
                # `_same_node` maps onto `run_gate_stage`'s `colocated`: the
                # CLI, unlike the pinned MCP server process, is only proven
                # colocated with whatever holds the `task_queue` "build"
                # lease when the operator passes --same-node. See
                # gate_runner.dispatch's retest docstring for the full
                # reasoning (this gates a HEAVY-tier Cargo build reservation,
                # not any decision this command's own ledger reads make).
                colocated = _same_node
                extra_job_data = {
                    "repo_name": repo_name,
                    "stage": stage,
                    "trigger": trigger,
                    "scope": scope,
                    "hook_ids_requested": list(hook_ids or []),
                    "colocated": colocated,
                }
                if _escalate_on_pass:
                    return job_store.submit_job(
                        "gate",
                        gate_runner.escalating_run_gate_stage,
                        path,
                        stage,
                        hook_ids,
                        timeout=600,
                        escalate_on_pass=True,
                        repo_name=repo_name,
                        submit_one=submit_one,
                        same_node=_same_node,
                        trigger=trigger,
                        scope=scope,
                        colocated=colocated,
                        record=True,
                        _extra_job_data=extra_job_data,
                    )
                return job_store.submit_job(
                    "gate",
                    run_gate_stage,
                    path,
                    stage,
                    timeout=600,
                    hook_ids=hook_ids,
                    trigger=trigger,
                    scope=scope,
                    colocated=colocated,
                    record=True,
                    _extra_job_data=extra_job_data,
                )

            return submit_one

        def _log_result(repo_name: str, job_id: str | None) -> bool:
            """Print one repo's completed gate result; True if it failed."""
            if not job_id:
                return False
            with job_store.jobs_lock:
                job = job_store.jobs.get(job_id)
            if job is None:
                return False
            result = job.get("result")
            if isinstance(result, RepoScanResult):
                runtime.logger.info(explain_gate_result(result))
                return not result.success
            if job.get("error"):
                runtime.logger.error(f"{repo_name}: {job['error']}")
                return True
            return False

        if args.gate:
            if _run_gate(
                runtime,
                args,
                gate_runner,
                _resolve_targets,
                _make_submit_one,
                _log_result,
            ):
                has_errors = True

        if args.gate_retest:
            if _run_gate_retest(
                runtime,
                args,
                gate_runner,
                default_gate_ledger,
                _resolve_targets,
                _make_submit_one,
                _log_result,
            ):
                has_errors = True

    return has_errors


def _run_gate(
    runtime: CliRuntime,
    args: argparse.Namespace,
    gate_runner: Any,
    resolve_targets: Any,
    make_submit_one: Any,
    log_result: Any,
) -> bool:
    has_errors = False
    run_result = gate_runner.dispatch(
        "run",
        resolve_targets=resolve_targets,
        submit_one=make_submit_one(args.gate),
        stage=args.gate,
        threads=args.threads,
        max_workers=args.threads,
    )
    if run_result.get("status") == "clean":
        runtime.logger.warning("No projects found for --gate.")
    else:
        runtime.logger.info(
            f"Running the {args.gate} gate across "
            f"{run_result['queued_count']} project(s) in parallel..."
        )
        failed_count = sum(
            1
            for repo_name, jid in run_result["jobs"].items()
            if log_result(repo_name, jid)
        )
        if failed_count:
            has_errors = True
            runtime.logger.error(
                f"{args.gate} gate failed in {failed_count}/"
                f"{len(run_result['jobs'])} project(s)."
            )
    return has_errors


def _run_gate_retest(
    runtime: CliRuntime,
    args: argparse.Namespace,
    gate_runner: Any,
    default_gate_ledger: Any,
    resolve_targets: Any,
    make_submit_one: Any,
    log_result: Any,
) -> bool:
    has_errors = False
    retest_result = gate_runner.dispatch(
        "retest",
        resolve_targets=resolve_targets,
        submit_one=make_submit_one(args.gate_retest),
        stage=args.gate_retest,
        threads=args.threads,
        max_workers=args.threads,
        gate_ledger=default_gate_ledger(),
        escalate=True,
        same_node=args.same_node,
    )
    runtime.logger.info(retest_result["message"])
    failed_count = 0
    for repo_name, entry in sorted(retest_result["targets"].items()):
        job_id = entry.get("retest_job_id")
        if not job_id:
            runtime.logger.info(
                f"{repo_name}: baseline={entry['baseline']}, nothing to retest."
            )
            continue
        runtime.logger.info(
            f"{repo_name}: baseline={entry['baseline']}, "
            f"hook_ids={entry['retest_hook_ids']}, stale={entry['stale']}"
        )
        if log_result(repo_name, job_id):
            failed_count += 1
    if failed_count:
        has_errors = True
        runtime.logger.error(
            f"--gate-retest found {failed_count}/"
            f"{len(retest_result['targets'])} project(s) still failing."
        )
    return has_errors


def _dispatch_validate(
    runtime: CliRuntime, git: object, args: argparse.Namespace, has_errors: bool
) -> bool:
    if not args.validate:
        return has_errors
    val_results = git.validate_and_release(
        threads=args.threads,
        auto_bump=bool(args.bump) if not args.maintain else False,
        auto_push=args.push,
        bump_part=args.bump if args.bump else "minor",
    )
    if not val_results.get("passed"):
        has_errors = True
        runtime.logger.error("Validation failed with errors.")
    else:
        runtime.logger.info(
            "Validation and subsequent operations completed successfully."
        )

    # Prevent these from executing again below
    args.bump = None
    args.push = False
    return has_errors


def _dispatch_bump(
    runtime: CliRuntime, git: object, args: argparse.Namespace, has_errors: bool
) -> bool:
    if not (args.bump and not args.maintain):
        return has_errors
    if has_errors and (args.push or args.bump):
        runtime.logger.error("Skipping bump due to preceding validation errors.")
        has_errors = True
    else:
        runtime.logger.info(f"Bumping version ({args.bump}) for all projects...")
        project_dirs = list(git.project_map.values())
        results = []
        for d in project_dirs:
            res = git.bump_version(
                args.bump, allow_dirty=True, path=d, dry_run=args.dry_run
            )
            results.append(res)
            if res.status == "error":
                has_errors = True

        summary = git.generate_markdown_summary("Bulk Version Bump", results)
        runtime.logger.info(summary)
        git._export_report(summary, "version_bump_report.md")
    return has_errors


def _load_config_file(runtime: CliRuntime, config_path: str) -> dict[str, Any] | None:
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        runtime.logger.error("Operation failed: error_type=%s", type(e).__name__)
        sys.exit(1)


def _dispatch_maintain(
    runtime: CliRuntime, git: object, args: argparse.Namespace, has_errors: bool
) -> bool:
    if not args.maintain:
        return has_errors
    if has_errors and (args.push or args.maintain):
        runtime.logger.error(
            "Skipping maintenance bump due to preceding validation errors."
        )
        has_errors = True
    else:
        config = _load_config_file(runtime, args.config) if args.config else None

        results = git.phased_bumpversion(
            part=args.bump if args.bump else "patch",
            start_phase=args.phase,
            dry_run=args.dry_run,
            allow_pre_commit=args.allow_pre_commit,
            config=config,
            single_phase=args.single_phase,
            project_filter=args.project,
            auto_start=args.auto_start,
        )

        for res in results:
            if res.status == "error":
                has_errors = True

        summary = git.generate_markdown_summary("Phased Maintenance Bump", results)

        runtime.logger.info(summary)
        git._export_report(summary, "maintenance_report.md")
    return has_errors


def _dispatch_push(
    runtime: CliRuntime, git: object, args: argparse.Namespace, has_errors: bool
) -> None:
    if not args.push:
        return
    if has_errors:
        runtime.logger.error(
            "Skipping push due to preceding validation or bump errors."
        )
    else:
        config = None
        if args.config:
            try:
                with open(args.config) as f:
                    config = json.load(f)
            except Exception as e:
                runtime.logger.error(
                    "Operation failed: error_type=%s", type(e).__name__
                )
                sys.exit(1)
        push_results = git.phased_push(
            start_phase=args.phase,
            config=config,
            single_phase=args.single_phase,
            project_filter=args.project,
            auto_start=args.auto_start,
        )
        summary = git.generate_markdown_summary("Phased Push", push_results)
        runtime.logger.info(summary)
        git._export_report(summary, "push_report.md")
