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

RM_GATES_ACTIONS = (
    "run",
    "status",
    "explain",
    "profile",
    "retest",
    # Configuration actions: they answer questions ABOUT the gates rather than
    # executing them, but a reader asking "why is this gate slow" or "can it
    # stop early" should not have to discover they live on a different tool.
    "audit_fail_fast",
    "xdist_plan",
    "xdist_apply",
)

RM_DOCS_READINESS_ACTIONS = (
    "preview",
    "apply",
    "verify",
)
