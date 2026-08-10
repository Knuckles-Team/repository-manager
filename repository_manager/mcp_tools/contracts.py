"""Stable action sets for the condensed Repository Manager MCP surface."""

RM_GIT_ACTIONS = (
    "raw",
    "clone",
    "enumerate",
    "pull",
    "push",
    "phased_push",
    "add",
    "commit",
    "pre_commit",
    "commit_code",
    "status",
    "cancel",
)

RM_WORKSPACE_ACTIONS = (
    "list",
    "list_branches",
    "setup",
    "template",
    "save",
    "maintain",
    "maintain_status",
)

RM_WORKTREE_ACTIONS = (
    "add",
    "list",
    "remove",
    "merge",
    "sync",
    "prune",
    "bulk_add",
    "audit",
    "reset_branch",
)

RM_MERGE_QUEUE_ACTIONS = (
    "enqueue",
    "status",
    "withdraw",
    "run",
    "config",
)

RM_BUILD_ACTIONS = (
    "request",
    "status",
    "artifacts",
    "explain",
    "gc",
)

RM_PROJECTS_ACTIONS = (
    "install",
    "build",
    "validate",
    "validate_status",
)
