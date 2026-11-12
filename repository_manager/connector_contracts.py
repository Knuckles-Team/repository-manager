"""Surface contracts: tool use, links, data locations and business roles."""

from scion_sdk.surface_contracts import (
    BusinessRole,
    DataLocation,
    Refusal,
    SurfaceLink,
    ToolStep,
    ToolUseContract,
)

CONTRACTS = (
    ToolUseContract(
        task="enumerate-and-ingest-repositories",
        description="Enumerate remote repositories of a forge and ingest them into the knowledge graph.",
        steps=(
            ToolStep(
                tool="rm_git_enumerate",
                params=("command", "path", "projects"),
                result=("repositories",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_ingest_repositories",
                params=("vcs", "scopes", "max_repos"),
                result=("ingested",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_workspace_list",
                params=("yml_path",),
                result=("projects",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="clone-repositories",
        description="Clone one repository or a set of workspace projects and read their status.",
        steps=(
            ToolStep(
                tool="repository_manager_clone_repository",
                params=("url", "target_path"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_clone_projects",
                params=("projects",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_clone",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_status",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("status", "jobs"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="sync-repositories",
        description="Pull one project or a set of projects and read the outcome.",
        steps=(
            ToolStep(
                tool="repository_manager_pull_project",
                params=("path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_pull_projects",
                params=("project_dirs",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_pull",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_status",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("status", "jobs"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="commit-and-push-project",
        description="Run pre-commit, commit and push one project.",
        steps=(
            ToolStep(
                tool="repository_manager_pre_commit",
                params=("run", "autoupdate", "path"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_commit_project",
                params=("message", "path"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_commit_code_project",
                params=("message", "run_precommit", "path"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_push_project",
                params=("path",),
                result=("status",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_git_action",
                params=("command", "path", "quiet", "env", "timeout", "raw_output"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="bulk-commit-and-push",
        description="Stage, gate, commit and push many projects as a tracked job, then poll or cancel it.",
        steps=(
            ToolStep(
                tool="rm_git_add",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_pre_commit",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_commit",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_commit_code",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_push",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("job_id",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_status",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("status", "jobs"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_git_cancel",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_pre_commit_projects",
                params=("run", "autoupdate", "projects"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_commit_projects",
                params=("message", "project_dirs"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_commit_code_projects",
                params=("message", "run_precommit", "project_dirs"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_push_projects",
                params=("project_dirs",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="refuse-raw-git",
        description="Show that a raw git command is retired and refused.",
        steps=(
            ToolStep(
                tool="rm_git_raw",
                params=(
                    "command",
                    "path",
                    "threads",
                    "projects",
                    "message",
                    "run_precommit",
                    "job_id",
                    "summary",
                ),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "RAW_COMMAND_RETIRED",
                        "raw host commands are retired; use a typed action",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="inspect-workspace",
        description="Read the workspace project list, branches, map and README.",
        steps=(
            ToolStep(
                tool="rm_workspace_list",
                params=("yml_path",),
                result=("projects",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_workspace_list_branches",
                params=("yml_path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_get_workspace_projects",
                params=(),
                result=("projects",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_get_project_map",
                params=(),
                result=("map",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_discover_projects",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_list_branches",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_get_readme",
                params=("path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_get_consolidated_skill_paths",
                params=(),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="configure-workspace",
        description="Set up the workspace from a manifest, template it, add or create projects and save the configuration.",
        steps=(
            ToolStep(
                tool="rm_workspace_template",
                params=("use_default",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_workspace_setup",
                params=("yml_path", "install"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_workspace_save",
                params=("yml_path", "config_dict"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_workspace_fleet_evidence_check",
                params=("yml_path", "fleet_evidence_json"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_generate_workspace_template",
                params=("target_path", "use_default"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_load_projects_from_yaml",
                params=("yaml_path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_setup_from_yaml",
                params=("yaml_path", "install"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_save_workspace_config",
                params=("yaml_path", "config"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_add_project",
                params=("path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_add_projects",
                params=("project_dirs",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_create_project",
                params=("path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_set_threads",
                params=("threads",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="install-build-and-test-projects",
        description="Install, build, test and validate workspace projects.",
        steps=(
            ToolStep(
                tool="rm_projects_install",
                params=("threads", "extra"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_projects_build",
                params=("threads", "repositories"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_projects_validate",
                params=("repositories", "failed_only"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_projects_validate_status",
                params=("job_id", "summary"),
                result=("status", "results"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_install_project",
                params=("path", "extra"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_install_projects",
                params=("extra", "threads", "report"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_build_projects",
                params=("threads",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_test_projects",
                params=("targets", "progress_phase", "progress_dict"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_validate_single_project",
                params=("repo_path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_validate_and_release",
                params=("threads",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_generate_markdown_summary",
                params=("operation", "results"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_cleanup_artifacts",
                params=("target_dir",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="bump-versions",
        description="Bump one project's version, bump the fleet, and update a dependency pin.",
        steps=(
            ToolStep(
                tool="repository_manager_bump_version",
                params=("part", "allow_dirty", "path", "dry_run", "verbose", "force"),
                result=("old_version", "new_version"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_bulk_bump",
                params=("part", "dry_run", "exclude", "verbose"),
                result=("results",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_update_dependency",
                params=("file_path", "package_name", "new_version", "dry_run"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="manage-worktrees",
        description="Open, list, audit, sync and merge isolated worktrees.",
        steps=(
            ToolStep(
                tool="rm_worktree_list",
                params=("repo",),
                result=("worktrees",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_add",
                params=("repo", "branch", "base", "into", "adopt"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_bulk_add",
                params=("repos", "branch", "base"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_sync",
                params=("repo", "branch", "strategy"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_merge",
                params=("repo", "branch", "into", "delete_branch"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_audit",
                params=("repo", "stale_days"),
                result=("worktrees",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="retire-worktrees",
        description="Remove, reset or prune worktrees after their work merged.",
        steps=(
            ToolStep(
                tool="rm_worktree_audit",
                params=("repo", "stale_days"),
                result=("worktrees",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_remove",
                params=("repo", "branch", "force", "delete_branch"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_reset_branch",
                params=("repo", "branch", "base", "force"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_worktree_prune",
                params=("repo", "prune_merged"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="repository_manager_worktree_hygiene",
                params=("prune", "base", "stale_days"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="run-merge-queue",
        description="Enqueue a branch, run the queue and read its status.",
        steps=(
            ToolStep(
                tool="rm_merge_queue_config",
                params=("repo_path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_merge_queue_enqueue",
                params=("repo_path", "branch", "base"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_merge_queue_status",
                params=("repo_path",),
                result=("queue", "state"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_merge_queue_run",
                params=("repo_path", "batch_size", "push"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_merge_queue_withdraw",
                params=("repo_path", "branch", "reason"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="coordinate-builds",
        description="Request a build, read its status, artifacts and explanation, and collect old artifacts.",
        steps=(
            ToolStep(
                tool="rm_build_request",
                params=("repo_path", "spec", "host"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_build_status",
                params=("repo_path", "key", "wait_timeout"),
                result=("state", "key"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_build_artifacts",
                params=("repo_path", "key"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_build_explain",
                params=("repo_path", "key"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_build_gc",
                params=("keep_recent", "max_age_days"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="run-fleet-gates",
        description="Run the gates over repositories, read status and explain failures; retest what failed.",
        steps=(
            ToolStep(
                tool="rm_gates_repos",
                params=("fleet",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_run",
                params=("fleet", "repos", "stage", "dry_run", "threads", "timeout"),
                result=("job_id",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_status",
                params=("job_id", "summary"),
                result=("status", "repos"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_explain",
                params=("job_id", "repo"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_retest",
                params=("job_id", "repo"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="tune-gate-execution",
        description="Profile gate runs, audit fail-fast, plan and apply parallel test runs and set phase direction.",
        steps=(
            ToolStep(
                tool="rm_gates_profile",
                params=("job_id", "top_n"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_audit_fail_fast",
                params=("repos", "escalate"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_xdist_plan",
                params=("repo",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_xdist_apply",
                params=("repo", "dry_run"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_phase_direction",
                params=("repos",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_gates_phase_direction_here",
                params=("repo",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="prepare-docs-readiness",
        description="Preview, apply and verify documentation readiness across a workspace.",
        steps=(
            ToolStep(
                tool="rm_docs_readiness_preview",
                params=("repository", "workspace", "manifest"),
                result=("changes",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="rm_docs_readiness_apply",
                params=("repository", "workspace", "manifest", "confirm"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
            ToolStep(
                tool="rm_docs_readiness_verify",
                params=("repository", "workspace", "manifest"),
                result=("ready",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                    Refusal(
                        "CONFIRMATION_REQUIRED",
                        "the caller has not confirmed the destructive call",
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="reserve-and-release-concepts",
        description="Reserve a concept name for a lane, read it, and release or reconcile the reservation.",
        steps=(
            ToolStep(
                tool="rm_concepts_list",
                params=("repo_root", "state", "concept_prefix", "limit", "cursor"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_get",
                params=("repo_root", "concept_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_reserve",
                params=(
                    "repo_root",
                    "tenant_ref",
                    "lane_ref",
                    "owner_ref",
                    "concept_id",
                    "namespace",
                    "purpose",
                ),
                result=("reservation_id", "fence"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_materialize",
                params=("repo_root", "reservation_id", "owner_ref", "expected_fence"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_release",
                params=("repo_root", "reservation_id", "owner_ref", "expected_fence"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_reconcile",
                params=("repo_root", "tenant_ref"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="certify-candidates",
        description="Verify a candidate or a generation of concepts against its source tree.",
        steps=(
            ToolStep(
                tool="rm_concepts_get",
                params=("repo_root", "concept_id"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_verify_candidate",
                params=("repo_root", "candidate", "source_tree_ish"),
                result=("verified",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_concepts_verify_generation",
                params=("repo_root", "generation", "candidates"),
                result=("verified",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="run-lane",
        description="Start a lane worktree, read its environment, diagnose and heal it, and finish it.",
        steps=(
            ToolStep(
                tool="rm_lane_start",
                params=("repo", "branch", "base"),
                result=("path", "branch"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_lane_env",
                params=("path",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_lane_doctor",
                params=("path",),
                result=("findings",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_lane_heal",
                params=("path", "force"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_lane_finish",
                params=("path", "force"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="operate-remote-workers",
        description="Register workers, seed them from the inventory, stage and verify source, dispatch a build and receive its artifact.",
        steps=(
            ToolStep(
                tool="rm_remote_workers_seed_from_inventory",
                params=("actor",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_register_worker",
                params=(
                    "host_id",
                    "cpu_weight",
                    "memory_mib",
                    "disk_mib",
                    "process_slots",
                    "labels",
                    "toolchains",
                ),
                result=("host_id", "state"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_profile",
                params=("host_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_recheck",
                params=("host_id",),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_stage_source",
                params=("host_id", "repository_id", "origin", "tree_sha"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_verify_source",
                params=("host_id", "repository_id", "expected_sha"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_dispatch_build",
                params=(
                    "host_id",
                    "repository_id",
                    "command",
                    "timeout_seconds",
                    "reservation_id",
                    "work_item_id",
                ),
                result=("attempt", "state"),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_receive_artifact",
                params=("host_id", "relative_path", "declared_digest", "kind"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_remote_workers_host_loss_reconcile",
                params=("host_id", "reason"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
    ToolUseContract(
        task="plan-and-roll-out-dependency",
        description="Plan a dependency rollout across the fleet, apply it, release and read its status.",
        steps=(
            ToolStep(
                tool="rm_rollout_plan",
                params=("dependency", "floor", "group", "only", "manifest"),
                result=("repositories",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_rollout_apply",
                params=("dependency", "floor", "only", "jobs", "dry_run"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_rollout_status",
                params=("dependency",),
                result=("state",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
            ToolStep(
                tool="rm_rollout_release",
                params=("dependency", "only", "dry_run"),
                result=("result",),
                refusals=(
                    Refusal(
                        "ACCESS_DENIED", "the credential lacks access to the operation"
                    ),
                    Refusal(
                        "INVALID_ARGUMENT", "the upstream system rejects the input"
                    ),
                ),
            ),
        ),
    ),
)

LINKS = (
    SurfaceLink(
        task="enumerate-and-ingest-repositories",
        tools=(
            "rm_git_enumerate",
            "repository_ingest_repositories",
            "rm_workspace_list",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=(
            "repository-manager-bulk-git-operations",
            "repository-manager-fleet-scale-operations",
        ),
    ),
    SurfaceLink(
        task="clone-repositories",
        tools=(
            "repository_manager_clone_repository",
            "repository_manager_clone_projects",
            "rm_git_clone",
            "rm_git_status",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=("repository-manager-bulk-git-operations",),
    ),
    SurfaceLink(
        task="sync-repositories",
        tools=(
            "repository_manager_pull_project",
            "repository_manager_pull_projects",
            "rm_git_pull",
            "rm_git_status",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=("repository-manager-bulk-git-operations",),
    ),
    SurfaceLink(
        task="commit-and-push-project",
        tools=(
            "repository_manager_pre_commit",
            "repository_manager_commit_project",
            "repository_manager_commit_code_project",
            "repository_manager_push_project",
            "repository_manager_git_action",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=("repository-manager-bulk-git-operations",),
    ),
    SurfaceLink(
        task="bulk-commit-and-push",
        tools=(
            "rm_git_add",
            "rm_git_pre_commit",
            "rm_git_commit",
            "rm_git_commit_code",
            "rm_git_push",
            "rm_git_status",
            "rm_git_cancel",
            "repository_manager_pre_commit_projects",
            "repository_manager_commit_projects",
            "repository_manager_commit_code_projects",
            "repository_manager_push_projects",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=(
            "repository-manager-bulk-git-operations",
            "repository-manager-fleet-scale-operations",
        ),
    ),
    SurfaceLink(
        task="refuse-raw-git",
        tools=("rm_git_raw",),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=("repository-manager-bulk-git-operations",),
    ),
    SurfaceLink(
        task="inspect-workspace",
        tools=(
            "rm_workspace_list",
            "rm_workspace_list_branches",
            "repository_manager_get_workspace_projects",
            "repository_manager_get_project_map",
            "repository_manager_discover_projects",
            "repository_manager_list_branches",
            "repository_manager_get_readme",
            "repository_manager_get_consolidated_skill_paths",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
            "repository-manager-release-engineer",
        ),
        skills=("repository-manager-operations",),
    ),
    SurfaceLink(
        task="configure-workspace",
        tools=(
            "rm_workspace_template",
            "rm_workspace_setup",
            "rm_workspace_save",
            "rm_workspace_fleet_evidence_check",
            "repository_manager_generate_workspace_template",
            "repository_manager_load_projects_from_yaml",
            "repository_manager_setup_from_yaml",
            "repository_manager_save_workspace_config",
            "repository_manager_add_project",
            "repository_manager_add_projects",
            "repository_manager_create_project",
            "repository_manager_set_threads",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=(
            "repository-manager-operations",
            "repository-manager-worker-operations",
        ),
    ),
    SurfaceLink(
        task="install-build-and-test-projects",
        tools=(
            "rm_projects_install",
            "rm_projects_build",
            "rm_projects_validate",
            "rm_projects_validate_status",
            "repository_manager_install_project",
            "repository_manager_install_projects",
            "repository_manager_build_projects",
            "repository_manager_test_projects",
            "repository_manager_validate_single_project",
            "repository_manager_validate_and_release",
            "repository_manager_generate_markdown_summary",
            "repository_manager_cleanup_artifacts",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=(
            "repository-manager-workspace-validation",
            "repository-manager-workspace-release",
        ),
    ),
    SurfaceLink(
        task="bump-versions",
        tools=(
            "repository_manager_bump_version",
            "repository_manager_bulk_bump",
            "repository_manager_update_dependency",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=("repository-manager-workspace-release",),
    ),
    SurfaceLink(
        task="manage-worktrees",
        tools=(
            "rm_worktree_list",
            "rm_worktree_add",
            "rm_worktree_bulk_add",
            "rm_worktree_sync",
            "rm_worktree_merge",
            "rm_worktree_audit",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=(
            "repository-manager-worktree-orchestration",
            "repository-manager-parallel-lane-orchestration",
        ),
    ),
    SurfaceLink(
        task="retire-worktrees",
        tools=(
            "rm_worktree_audit",
            "rm_worktree_remove",
            "rm_worktree_reset_branch",
            "rm_worktree_prune",
            "repository_manager_worktree_hygiene",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=("repository-manager-worktree-orchestration",),
    ),
    SurfaceLink(
        task="run-merge-queue",
        tools=(
            "rm_merge_queue_config",
            "rm_merge_queue_enqueue",
            "rm_merge_queue_status",
            "rm_merge_queue_run",
            "rm_merge_queue_withdraw",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=(
            "repository-manager-merge-and-reconcile",
            "repository-manager-parallel-lane-orchestration",
        ),
    ),
    SurfaceLink(
        task="coordinate-builds",
        tools=(
            "rm_build_request",
            "rm_build_status",
            "rm_build_artifacts",
            "rm_build_explain",
            "rm_build_gc",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=("repository-manager-build-coordination",),
    ),
    SurfaceLink(
        task="run-fleet-gates",
        tools=(
            "rm_gates_repos",
            "rm_gates_run",
            "rm_gates_status",
            "rm_gates_explain",
            "rm_gates_retest",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=(
            "repository-manager-gate-execution",
            "repository-manager-candidate-certification",
        ),
    ),
    SurfaceLink(
        task="tune-gate-execution",
        tools=(
            "rm_gates_profile",
            "rm_gates_audit_fail_fast",
            "rm_gates_xdist_plan",
            "rm_gates_xdist_apply",
            "rm_gates_phase_direction",
            "rm_gates_phase_direction_here",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=("repository-manager-gate-execution",),
    ),
    SurfaceLink(
        task="prepare-docs-readiness",
        tools=(
            "rm_docs_readiness_preview",
            "rm_docs_readiness_apply",
            "rm_docs_readiness_verify",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=(
            "repository-manager-workspace-release",
            "repository-manager-fleet-scale-operations",
        ),
    ),
    SurfaceLink(
        task="reserve-and-release-concepts",
        tools=(
            "rm_concepts_list",
            "rm_concepts_get",
            "rm_concepts_reserve",
            "rm_concepts_materialize",
            "rm_concepts_release",
            "rm_concepts_reconcile",
        ),
        prompts=("repository-manager-operator", "repository-manager-concept-steward"),
        skills=("repository-manager-concept-coordination",),
    ),
    SurfaceLink(
        task="certify-candidates",
        tools=(
            "rm_concepts_get",
            "rm_concepts_verify_candidate",
            "rm_concepts_verify_generation",
        ),
        prompts=("repository-manager-operator", "repository-manager-concept-steward"),
        skills=(
            "repository-manager-candidate-certification",
            "repository-manager-concept-coordination",
        ),
    ),
    SurfaceLink(
        task="run-lane",
        tools=(
            "rm_lane_start",
            "rm_lane_env",
            "rm_lane_doctor",
            "rm_lane_heal",
            "rm_lane_finish",
        ),
        prompts=(
            "repository-manager-operator",
            "repository-manager-repository-maintainer",
        ),
        skills=(
            "repository-manager-lane-lifecycle",
            "repository-manager-parallel-lane-orchestration",
            "repository-manager-development-lifecycle",
        ),
    ),
    SurfaceLink(
        task="operate-remote-workers",
        tools=(
            "rm_remote_workers_seed_from_inventory",
            "rm_remote_workers_register_worker",
            "rm_remote_workers_profile",
            "rm_remote_workers_recheck",
            "rm_remote_workers_stage_source",
            "rm_remote_workers_verify_source",
            "rm_remote_workers_dispatch_build",
            "rm_remote_workers_receive_artifact",
            "rm_remote_workers_host_loss_reconcile",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=("repository-manager-worker-operations",),
    ),
    SurfaceLink(
        task="plan-and-roll-out-dependency",
        tools=(
            "rm_rollout_plan",
            "rm_rollout_apply",
            "rm_rollout_status",
            "rm_rollout_release",
        ),
        prompts=("repository-manager-operator", "repository-manager-release-engineer"),
        skills=(
            "repository-manager-workspace-release",
            "repository-manager-fleet-scale-operations",
        ),
    ),
)

LOCATIONS = (
    DataLocation(
        name="workspace-projects",
        holds=("GitRepository", "Project"),
        instance="REPOSITORY_MANAGER_WORKSPACE",
        read_tool="repository_manager_get_workspace_projects",
        federation_key="fullPath",
    ),
    DataLocation(
        name="worktrees",
        holds=("Worktree",),
        instance="REPOSITORY_MANAGER_WORKSPACE",
        read_tool="rm_worktree_list",
        federation_key="path",
    ),
)

ROLES = (
    BusinessRole(
        name="repository maintainer",
        description="Keep the workspace projects cloned, current, committed and pushed on isolated worktrees.",
        tasks=(
            "enumerate-and-ingest-repositories",
            "clone-repositories",
            "sync-repositories",
            "commit-and-push-project",
            "bulk-commit-and-push",
            "refuse-raw-git",
            "inspect-workspace",
            "configure-workspace",
            "manage-worktrees",
            "retire-worktrees",
            "run-merge-queue",
            "run-lane",
        ),
    ),
    BusinessRole(
        name="release engineer",
        description="Validate, build, gate, version and roll out the fleet.",
        tasks=(
            "install-build-and-test-projects",
            "bump-versions",
            "coordinate-builds",
            "run-fleet-gates",
            "tune-gate-execution",
            "prepare-docs-readiness",
            "plan-and-roll-out-dependency",
            "operate-remote-workers",
            "inspect-workspace",
        ),
    ),
    BusinessRole(
        name="concept steward",
        description="Reserve, materialize, verify and release concept names for parallel lanes without collisions.",
        tasks=("reserve-and-release-concepts", "certify-candidates"),
    ),
)
