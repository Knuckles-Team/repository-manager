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

from repository_manager.cli_commands.build_queue import run_build_queue_cli
from repository_manager.cli_commands.context import CliRuntime
from repository_manager.cli_commands.lane import run_lane_cli
from repository_manager.cli_commands.merge_queue import run_merge_queue_cli


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
        choices=["doctor", "start", "finish", "env"],
        help=(
            "'doctor' checks this tree's isolation and mutates nothing; 'start' "
            "opens an isolated worktree and proves its partitions; 'env' prints "
            "the shell exports; 'finish' preflights then enqueues the branch for "
            "landing. Run 'doctor' whenever something behaves impossibly."
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

    args = parser.parse_args()

    # Handled before every other verb and returns immediately: the queue drives a
    # SINGLE named repository and must not be combined with the workspace-wide
    # bulk operations above, whose --path/--repositories semantics are different.
    if args.merge_queue:
        sys.exit(run_merge_queue_cli(args))
    if args.build_broker:
        sys.exit(run_build_queue_cli(args))

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
    clone_flag = args.clone
    pull_flag = args.pull

    if args.default_branch:
        git.set_to_default_branch = True

    if args.threads:
        git.set_threads(threads=args.threads)

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

    if args.repositories:
        repositories = args.repositories.replace(" ", "").split(",")
        names_to_keep = set(repositories)
        if git.project_map:
            filtered = {}
            for url, path in git.project_map.items():
                name = url.split("/")[-1].replace(".git", "")
                if name in names_to_keep:
                    filtered[url] = path
            git.project_map = filtered
        else:
            for r in repositories:
                if "/" in r:
                    name = r.split("/")[-1].replace(".git", "")
                    git.project_map[r] = os.path.join(git.path, name)
                else:
                    git.project_map[os.path.join("https://github.com/", r)] = (
                        os.path.join(git.path, r)
                    )

    if args.file and os.path.exists(args.file):
        if args.setup:
            runtime.logger.info("Setting up workspace from configured manifest")
            git.load_projects_from_yaml(args.file)

    if clone_flag:
        git.clone_projects()
    if pull_flag:
        git.pull_projects()

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

    has_errors = False

    if args.validate:
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

    if args.bump and not args.maintain:
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

    if args.maintain:
        if has_errors and (args.push or args.maintain):
            runtime.logger.error(
                "Skipping maintenance bump due to preceding validation errors."
            )
            has_errors = True
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

    if args.push:
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
