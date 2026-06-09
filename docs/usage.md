# Usage — API / CLI / MCP

`repository-manager` exposes the same capability three ways: as **MCP tools** an
agent calls, as a **Python API** (`Git`) you import, and as a **command-line
interface**.

## As an MCP server

Once [deployed](deployment.md), the server registers consolidated, action-routed tool
modules. Each module groups related methods behind one tool to keep the LLM context
small, and each can be toggled independently with its environment variable.

| Module | Toggle | Default | Action-routed methods |
|---|---|---|---|
| Misc | `MISCTOOL` | `True` | health check and miscellaneous helpers |
| Git Operations | `GIT_OPERATIONSTOOL` | `True` | `clone`, `pull`, `push`, `phased_push`, `raw` |
| Workspace Management | `WORKSPACE_MANAGEMENTTOOL` | `True` | `list`, `list_branches`, `maintain`, `remediate`, `save`, `setup`, `template` |
| Project Management | `PROJECT_MANAGEMENT_TOOL` | `True` | `build`, `install`, `validate`, `validate_status` |

Example agent prompts that map onto these tools:

- *"List every project in the workspace."* → `workspace_management list`
- *"Pull the latest changes for all repositories."* → `git_operations pull`
- *"Validate the workspace, then run a phased push of the agents."* → `project_management validate` + `git_operations phased_push`

## As a Python API

`Git` (`repository_manager.repository_manager`) is a workspace-aware client for bulk
Git operations and workspace introspection.

```python
from repository_manager.repository_manager import Git

git = Git(path="/home/apps/workspace")

# Reads
projects = git.get_workspace_projects()        # list of managed project names
project_map = git.get_project_map()            # name -> absolute path
branches = git.list_branches()                 # name -> current branch

# Bulk operations
git.pull_projects()                            # pull every managed repository
git.clone_projects(["agent-utilities"])        # clone selected projects

# Validation
result = git.validate_single_project(project_map["agent-utilities"])
```

Load a workspace from its declarative `workspace.yml`:

```python
git = Git(path="/home/apps/workspace")
git.setup_from_yaml("workspace.yml")           # materialize the declared estate
```

## As a CLI

The `repository-manager` console script drives the full maintenance lifecycle from
the command line.

```bash
# Set up the workspace from its declared configuration
repository-manager --setup

# Enumerate branches across every managed repository
repository-manager --branches

# Clone and pull in bulk
repository-manager --clone
repository-manager --pull
```

The autonomous release harness runs a validation → bump → maintain → push sequence
that aborts on the first failure:

```bash
repository-manager --validate --bump patch --maintain --push
```

- **`--validate`** runs a full pre-release validation; subsequent steps abort on failure.
- **`--bump [patch|minor|major]`** bumps semantic versions.
- **`--maintain`** propagates version changes through the dependency tree.
- **`--push`** runs a parallelized, phase-gated Git push respecting `wait_minutes`.

The phased mechanics are documented in detail in
[Phased Maintenance](phased_maintenance.md) and [Phased Push](phased_push.md).
