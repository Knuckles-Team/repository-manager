# Adapter boundaries

Repository Manager keeps its public MCP and CLI surfaces stable while allowing
their implementation to evolve in small, reviewable modules.  RMDD-04 moved
the FastMCP registrations and the command-line parser out of the two historical
monoliths without moving policy into the adapters.

## MCP surface

`repository_manager/mcp_server.py` remains the compatibility entrypoint.  It
owns server startup and the shared background-job helpers.  The registration
surface is supplied by `repository_manager/mcp_tools/registry.py` in this
fixed order:

1. `git_operations` — `GIT_OPERATIONSTOOL`
2. `misc` — `MISCTOOL`
3. `project_management` — `PROJECT_MANAGEMENTTOOL`
4. `workspace_management` — `WORKSPACE_MANAGEMENTTOOL`

The registry is passed explicitly to `register_tool_surface`, so tool discovery
does not depend on incidental module names or import order.  The project
registrar deliberately retains the historical nested order: lane, worktree,
merge queue, build, then bulk projects.  The condensed catalog, tool names,
parameter schemas, descriptions, tags, defaults, and required fields are
covered by `tests/test_adapter_extraction.py` against a baseline digest.

Each domain adapter only performs FastMCP declaration and argument marshalling:

| Adapter | Existing core it delegates to |
| --- | --- |
| `mcp_tools/git.py` | `Git` and the job helpers exposed by `mcp_server` |
| `mcp_tools/misc.py` | VCS enumeration and KG ingestion |
| `mcp_tools/lane.py` | `lane_doctor` |
| `mcp_tools/worktree.py` | `WorktreeManager` |
| `mcp_tools/merge_queue.py` | merge-queue dispatch |
| `mcp_tools/build.py` | build-queue dispatch |
| `mcp_tools/projects.py` | project install/build/validation cores |
| `mcp_tools/workspace.py` | workspace configuration and maintenance |

`McpToolContext` resolves the server-owned helpers at call time.  This keeps
the existing test/deployment seams (including monkeypatches and configured
runtime instances) valid after registration has happened, and avoids import-
time work or duplicated job state.

Future MCP domains should add a registrar and compose it with
`mcp_tools.extend_registry(...)`; they should not edit `mcp_server.py` or
reintroduce module-name discovery.  An extension must preserve the existing
four-entry order and must add an explicit environment gate before deployment.
No new domain or tool is enabled by RMDD-04.

## CLI surface

`repository_manager/repository_manager.py` remains the packaged entrypoint and
compatibility export for `main`, the queue runners, `Git`, and manifest helpers.
`repository_manager/cli_commands/parser.py` owns argument construction and
dispatch marshalling through a `CliRuntime`.  The queue and lane subcommands
are isolated in `cli_commands/{build_queue,merge_queue,lane}.py`.

The runtime object is late-bound from the canonical module.  Consequently,
existing callers that patch `repository_manager.repository_manager.Git`,
configuration defaults, or manifest synchronization continue to exercise the
same core behavior.  New CLI verbs should be placed in `cli_commands/`, receive
their dependencies through `CliRuntime`, and leave the root module as a thin
entrypoint shim.

## Validation levels

This extraction changes structure only.  The fast lane gate is therefore the
adapter contract test plus the focused MCP registration/handler and CLI tests.
The repository's normal pre-commit suite remains the authoritative full gate;
it is run with the lane environment exports so caches and pytest temporary
directories are partitioned.  RMDD-04 introduces no dependency, persistence,
security, resource-policy, tool, action, or task changes.
