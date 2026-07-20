# Repository Manager
## CLI or API | MCP | Agent

![PyPI - Version](https://img.shields.io/pypi/v/repository-manager)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/repository-manager)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/repository-manager)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/repository-manager)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/repository-manager)
![PyPI - License](https://img.shields.io/pypi/l/repository-manager)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/repository-manager)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/repository-manager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/repository-manager)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/repository-manager)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/repository-manager)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/repository-manager)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/repository-manager)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/repository-manager)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/repository-manager)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/repository-manager)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/repository-manager)

*Version: 2.0.1*

> **Documentation** — Installation, deployment, usage across the CLI, API, MCP, and
> agent interfaces, and the phased multi-repository release workflows are maintained
> in the [official documentation](https://knuckles-team.github.io/repository-manager/).

---

## Overview

**Repository Manager** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Manage your git projects.

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.

---

## CLI or API

This agent wraps the Manage your git projects API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

This table is auto-generated from the live server — do not edit by hand.

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `rm_git` | `GIT_OPERATIONSTOOL` | Bulk Git operations and arbitrary command execution. |
| `rm_projects` | `PROJECT_MANAGEMENTTOOL` | Bulk install, build, and validate Python projects. |
| `rm_workspace` | `WORKSPACE_MANAGEMENTTOOL` | Core workspace organization, configuration, and maintenance. |
| `rm_worktree` | `PROJECT_MANAGEMENTTOOL` | Manage git worktrees for concurrent multi-session development (CONCEPT:RM-WORKTREE). |

#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)

<details>
<summary>42 per-operation tools — one per public API method (click to expand)</summary>

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `repository_manager_add_project` | `GITTOOL` | Stage all changes (git add -A) for a single Git project. |
| `repository_manager_add_projects` | `GITTOOL` | Stage all changes for multiple projects in parallel. |
| `repository_manager_build_projects` | `GITTOOL` | Bulk builds Python and Node.js projects in the workspace. |
| `repository_manager_bulk_bump` | `GITTOOL` | Bumps the version for all projects in the workspace in parallel. |
| `repository_manager_bump_version` | `GITTOOL` | Bump the version of the project using bump2version. |
| `repository_manager_cleanup_artifacts` | `GITTOOL` | Removes test artifacts and temporary files from the specified directory. |
| `repository_manager_clone_projects` | `GITTOOL` | Clone all specified Git projects in parallel using multiple threads. |
| `repository_manager_clone_repository` | `GITTOOL` | Clone a single Git repository to a specific target path. |
| `repository_manager_commit_code_project` | `GITTOOL` | Stage ALL changes (git add -A), optionally gate on pre-commit, then commit. |
| `repository_manager_commit_code_projects` | `GITTOOL` | Concurrently stage + pre-commit + commit feature code across projects. |
| `repository_manager_commit_project` | `GITTOOL` | Commit staged changes (git commit -m "{message}") for a single Git project. |
| `repository_manager_commit_projects` | `GITTOOL` | Commit staged changes for multiple projects in parallel. |
| `repository_manager_create_project` | `GITTOOL` | Create a new project directory and initialize it as a git repository. |
| `repository_manager_discover_projects` | `GITTOOL` | Scan self.path for immediate subdirectories containing a .git folder. |
| `repository_manager_generate_markdown_summary` | `GITTOOL` | Generates a beautiful markdown summary of bulk operation results. |
| `repository_manager_generate_workspace_template` | `GITTOOL` | Generates a workspace.yml template at the specified path. |
| `repository_manager_get_consolidated_skill_paths` | `GITTOOL` | Returns absolute paths to the 15 specific building and documentation skills. |
| `repository_manager_get_project_map` | `GITTOOL` | Returns the mapping of repository URLs to their local project paths. |
| `repository_manager_get_readme` | `GITTOOL` | Get the content and path of the README.md file in the specified path. |
| `repository_manager_get_workspace_projects` | `GITTOOL` | Returns a list of project basenames (e.g. 'genius-agent') defined in the workspace. |
| `repository_manager_git_action` | `GITTOOL` | Execute a Git command in the specified directory. |
| `repository_manager_install_project` | `GITTOOL` | Install a Python project using pip install -e .[extra]. |
| `repository_manager_install_projects` | `GITTOOL` | Bulk installs Python and Node projects in the workspace. |
| `repository_manager_list_branches` | `GITTOOL` | Returns a dictionary mapping project basenames to their current active git branch. |
| `repository_manager_load_projects_from_yaml` | `GITTOOL` | Loads repository URLs from a YAML workspace file using Pydantic models. |
| `repository_manager_maintain_projects` | `GITTOOL` | Execute the phased bumpversion workflow: pre-commits + phased bumping. |
| `repository_manager_phased_bumpversion` | `GITTOOL` | Execute the phased bumpversion workflow: pre-commits + phased bumping. |
| `repository_manager_phased_push` | `GITTOOL` | Execute the phased git push workflow. |
| `repository_manager_pre_commit` | `GITTOOL` | Execute pre-commit commands in the specified path. |
| `repository_manager_pre_commit_projects` | `GITTOOL` | Execute pre-commit commands for all projects in parallel. |
| `repository_manager_pull_project` | `GITTOOL` | Pull updates for a single Git project and optionally checkout the default branch. |
| `repository_manager_pull_projects` | `GITTOOL` | Pull updates for multiple projects in parallel. |
| `repository_manager_push_project` | `GITTOOL` | Push updates and tags for a single Git project, ensuring all staged and unstaged changes are committed first. |
| `repository_manager_push_projects` | `GITTOOL` | Push updates for multiple projects in parallel. |
| `repository_manager_save_workspace_config` | `GITTOOL` | Saves the current or provided WorkspaceConfig to a YAML file. |
| `repository_manager_set_threads` | `GITTOOL` | Set the number of threads for parallel processing. |
| `repository_manager_setup_from_yaml` | `GITTOOL` | Sets up the workspace structure from a YAML file. |
| `repository_manager_test_projects` | `GITTOOL` | Execute pytests for the specified projects in parallel. |
| `repository_manager_update_dependency` | `GITTOOL` | Update a package's pinned version in a deps file (pyproject OR requirements). |
| `repository_manager_validate_and_release` | `GITTOOL` | Validate projects in parallel, optionally triggering a release if successful. |
| `repository_manager_validate_single_project` | `GITTOOL` | Validates a single repository by running the scanner logic. |
| `repository_manager_worktree_hygiene` | `GITTOOL` | Audit (and optionally prune) session worktrees as a release-flow step. |

</details>

_4 action-routed tool(s) (default) · 42 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/usage.md](docs/usage.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the connector-focused `[mcp]` extra.** Examples use `repository-manager[mcp]` to add
> FastMCP / FastAPI through `agent-utilities[mcp]`; the required Agent Utilities core
> still carries `epistemic-graph[full]`. The `[agent-runtime]` extra additionally
> enables model orchestration.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

```json
{
  "mcpServers": {
    "repository-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "repository-manager[mcp]",
        "repository-manager-mcp"
      ],
      "env": {
        "MCP_TOOL_MODE": "intent",
        "GH_TOKEN": "your_github_token_here",
        "GITHUB_TOKEN": "your_github_token_here",
        "GITLAB_HOST": "https://gitlab.com",
        "GITLAB_PRIVATE_TOKEN": "your_gitlab_token_here",
        "GITLAB_TOKEN": "your_gitlab_token_here",
        "GITLAB_URL": "https://gitlab.com",
        "GIT_OPERATIONSTOOL": "True",
        "MISCTOOL": "True",
        "PROJECT_MANAGEMENTTOOL": "True",
        "REPOSITORY_MANAGER_DEFAULT_BRANCH": "main",
        "REPOSITORY_MANAGER_THREADS": "12",
        "RM_GATE_BEFORE_PUSH": "true",
        "RM_JOB_STALE_SECONDS": "1800",
        "RM_MAX_WORKERS": "8",
        "WORKSPACE_MANAGEMENTTOOL": "True",
        "WORKSPACE_YML": "workspace.yml"
      }
    }
  }
}
```

Runtime references require an alias-aware launcher such as GraphOS. Other
launchers must omit those entries and inject the resolved values through their
own runtime secret boundary.

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "repository-manager-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "repository-manager[mcp]",
        "repository-manager-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "MCP_TOOL_MODE": "intent",
        "GH_TOKEN": "your_github_token_here",
        "GITHUB_TOKEN": "your_github_token_here",
        "GITLAB_HOST": "https://gitlab.com",
        "GITLAB_PRIVATE_TOKEN": "your_gitlab_token_here",
        "GITLAB_TOKEN": "your_gitlab_token_here",
        "GITLAB_URL": "https://gitlab.com",
        "GIT_OPERATIONSTOOL": "True",
        "MISCTOOL": "True",
        "PROJECT_MANAGEMENTTOOL": "True",
        "REPOSITORY_MANAGER_DEFAULT_BRANCH": "main",
        "REPOSITORY_MANAGER_THREADS": "12",
        "RM_GATE_BEFORE_PUSH": "true",
        "RM_JOB_STALE_SECONDS": "1800",
        "RM_MAX_WORKERS": "8",
        "WORKSPACE_MANAGEMENTTOOL": "True",
        "WORKSPACE_YML": "workspace.yml"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:

```json
{
  "mcpServers": {
    "repository-manager-mcp": {
      "url": "http://localhost:8000/repository-manager-mcp/mcp"
    }
  }
}
```

Run a reviewed container image as a least-privilege stdio child (no
listener or published port):

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  -e MCP_TOOL_MODE=intent \
  -e GH_TOKEN=your_github_token_here \
  -e GITHUB_TOKEN=your_github_token_here \
  -e GITLAB_HOST=https://gitlab.com \
  -e GITLAB_PRIVATE_TOKEN=your_gitlab_token_here \
  -e GITLAB_TOKEN=your_gitlab_token_here \
  -e GITLAB_URL=https://gitlab.com \
  -e GIT_OPERATIONSTOOL=True \
  -e MISCTOOL=True \
  -e PROJECT_MANAGEMENTTOOL=True \
  -e REPOSITORY_MANAGER_DEFAULT_BRANCH=main \
  -e REPOSITORY_MANAGER_THREADS=12 \
  -e RM_GATE_BEFORE_PUSH=true \
  -e RM_JOB_STALE_SECONDS=1800 \
  -e RM_MAX_WORKERS=8 \
  -e WORKSPACE_MANAGEMENTTOOL=True \
  -e WORKSPACE_YML=workspace.yml \
  registry.example.invalid/repository-manager@sha256:<digest> repository-manager-mcp
```

For containerized network HTTP, supply an authenticated TLS ingress (or
direct server TLS), exact `MCP_ALLOWED_HOSTS`, and an exact trusted-proxy
CIDR policy through the operator-owned deployment profile. The generator
does not emit an unauthenticated non-loopback listener.

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`repository-manager` can run as a local stdio process or container, or behind a remote
network boundary. The
[Deployment guide](https://knuckles-team.github.io/repository-manager/deployment/) carries
the detailed transport contract.

- **Local container** — launch a reviewed immutable image as a least-privilege
  stdio child with no listener or published port.
- **Remote URL** — connect through an operator-supplied authenticated HTTPS
  ingress. Keep its URL, outbound identity references, trust profile, and exact
  `MCP_ALLOWED_HOSTS` in `AgentConfig`.
<!-- END GENERATED: additional-deployment-options -->

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set workspace + git operation settings
export REPOSITORY_MANAGER_WORKSPACE="your_value"
export WORKSPACE_YML="workspace.yml"
export REPOSITORY_MANAGER_DEFAULT_BRANCH="main"
export REPOSITORY_MANAGER_THREADS="12"
# Optional VCS provider credentials
export GITLAB_TOKEN="your_gitlab_token"
export GITHUB_TOKEN="your_github_token"

# Run the agent server
repository-manager-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  repository-manager-mcp:
    image: example/repository-manager:mcp
    container_name: repository-manager-mcp
    hostname: repository-manager-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  repository-manager-agent:
    image: example/repository-manager@sha256:<digest>
    container_name: repository-manager-agent
    hostname: repository-manager-agent
    restart: always
    depends_on:
      - repository-manager-mcp
    env_file:
      - ../.env
    command: [ "repository-manager-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9047
      - MCP_URL=http://repository-manager-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9047:9047"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9047/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/deployment.md](docs/deployment.md).

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `AGENT_UTILITIES_WORKSPACE_ROOT` | — | Runtime-injected ecosystem root; never persist a machine path |
| `AGENT_UTILITIES_REPO_ORIGIN` | — | Private Git origin used by the portable workspace manifest |
| `AGENT_UTILITIES_SERVICE_DOMAIN_SUFFIX` | `example.invalid` | Deployment DNS suffix used by service metadata |
| `REPOSITORY_MANAGER_WORKSPACE` | — | Optional workspace-root override |
| `WORKSPACE_PATH` | — | Compatibility workspace-root override |
| `WORKSPACE_YML` | `workspace.yml` | workspace manifest filename (resolved under the workspace root) |
| `WORKSPACE_REPORTS` | — | Optional runtime report directory |
| `REPOSITORY_MANAGER_WORKTREE_ROOT` | — | Optional runtime worktree root |
| `REPOSITORY_MANAGER_DEFAULT_BRANCH` | `main` | default branch name for git operations |
| `REPOSITORY_MANAGER_THREADS` | `12` | parallel threads for git operations |
| `RM_MAX_WORKERS` | `8` | explicit worker-count override (skips auto-sizing from CPU count) |
| `RM_JOB_STALE_SECONDS` | `1800` | seconds before an in-flight job is treated as stale by the watchdog |
| `RM_GATE_BEFORE_PUSH` | `true` | run the pre-commit gate before pushing (set false to bypass) |
| `GITLAB_URL` | `https://gitlab.com` | GitLab base URL (alias: GITLAB_HOST) |
| `GITLAB_HOST` | `https://gitlab.com` | legacy alias for GITLAB_URL |
| `GITLAB_TOKEN` | `your_gitlab_token_here` | GitLab access token (alias: GITLAB_PRIVATE_TOKEN) |
| `GITLAB_PRIVATE_TOKEN` | `your_gitlab_token_here` | legacy alias for GITLAB_TOKEN |
| `GITHUB_TOKEN` | `your_github_token_here` | GitHub access token (alias: GH_TOKEN) |
| `GH_TOKEN` | `your_github_token_here` | legacy alias for GITHUB_TOKEN |
| `GIT_OPERATIONSTOOL` | `True` | MCP tools table (condensed action-routed surface). |
| `PROJECT_MANAGEMENTTOOL` | `True` |  |
| `WORKSPACE_MANAGEMENTTOOL` | `True` |  |
| `MISCTOOL` | `True` |  |
| `AUTH_TYPE` | `bearer` | authentication type (e.g. bearer, none) |
| `LLM_API_KEY` | `your_llm_api_key_here` |  |
| `LLM_BASE_URL` | `https://api.openai.com/v1` |  |
| `MCP_URL` | `http://localhost:8000` |  |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_35 package + 13 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server and agent read, sourced from [`.env.example`](.env.example).

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_TOOL_MODE` | Tool surface: `condensed`, `verbose`, or `both` | `condensed` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |
| `PYTHONUNBUFFERED` | Unbuffered stdout (recommended in containers) | `1` |

### Workspace & repository management
| Variable | Description | Default |
|----------|-------------|---------|
| `REPOSITORY_MANAGER_WORKSPACE` | Root directory of the git workspace to manage | deployment-provided |
| `WORKSPACE_PATH` | Fallback workspace root when `REPOSITORY_MANAGER_WORKSPACE` is unset | deployment-provided |
| `WORKSPACE_YML` | Workspace manifest filename (resolved under the workspace root) | `workspace.yml` |
| `WORKSPACE_REPORTS` | Directory where generated reports are written | — |
| `REPOSITORY_MANAGER_WORKTREE_ROOT` | Root directory for managed git worktrees | deployment-provided |
| `REPOSITORY_MANAGER_DEFAULT_BRANCH` | Default branch name for git operations | `main` |
| `REPOSITORY_MANAGER_THREADS` | Parallel threads for git operations | `12` |

### Operations & gating
| Variable | Description | Default |
|----------|-------------|---------|
| `RM_MAX_WORKERS` | Explicit worker-count override (skips auto-sizing from CPU count) | — |
| `RM_JOB_STALE_SECONDS` | Seconds before an in-flight job is treated as stale by the watchdog | `1800` |
| `RM_GATE_BEFORE_PUSH` | Run the pre-commit gate before pushing (set `false` to bypass) | `true` |

### Version-control provider credentials
| Variable | Description | Default |
|----------|-------------|---------|
| `GITLAB_URL` | GitLab base URL (alias: `GITLAB_HOST`) | — |
| `GITLAB_TOKEN` | GitLab access token (alias: `GITLAB_PRIVATE_TOKEN`) | — |
| `GITHUB_TOKEN` | GitHub access token (alias: `GH_TOKEN`) | — |

### Tool toggles
Each action-routed tool can be disabled individually via its toggle env var (set to `false`).
The full list is in the [Available MCP Tools](#available-mcp-tools) table above.

| Variable | Description | Default |
|----------|-------------|---------|
| `GIT_OPERATIONSTOOL` | Enable bulk Git operations (`rm_git`) | `True` |
| `PROJECT_MANAGEMENTTOOL` | Enable project + worktree management (`rm_projects` / `rm_worktree`) | `True` |
| `WORKSPACE_MANAGEMENTTOOL` | Enable workspace organization (`rm_workspace`) | `True` |

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

### Agent CLI (full `[agent]` runtime only)
| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_URL` | URL of the MCP server the agent connects to | `http://localhost:8000/mcp` |
| `PROVIDER` | LLM provider (e.g. `openai`) | `openai` |
| `MODEL_ID` | Model id (e.g. `gpt-4o`) | `gpt-4o` |
| `ENABLE_WEB_UI` | Serve the AG-UI web interface | `True` |

---

## Security & Governance

Built directly upon the enterprise-ready [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Functionality | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Installation

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `repository-manager[mcp]` | Connector-focused MCP server (`agent-utilities[mcp]` — FastMCP/FastAPI + `epistemic-graph[full]`) | You only run the **MCP server** (smallest install / image) |
| `repository-manager[agent]` | Full agent runtime (`agent-utilities[agent-runtime,logfire]` — Pydantic AI + the epistemic-graph engine, plus `pre-commit` / `bump2version`) | You run the **integrated agent** |
| `repository-manager[all]` | Everything (`mcp` + `agent` + `logfire`) | Development / both surfaces |

```bash
# Connector-focused MCP server (includes the shared graph engine)
uv pip install "repository-manager[mcp]"

# Agent runtime (adds model orchestration to the shared graph engine)
uv pip install "repository-manager[agent]"

# Everything (development)
uv pip install "repository-manager[all]"      # or: python -m pip install "repository-manager[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `example/repository-manager:mcp` | `--target mcp` | `repository-manager[mcp]` — **connector-focused**, includes `epistemic-graph[full]`; no model-orchestration stack | `repository-manager-mcp` |
| `example/repository-manager@sha256:<digest>` | `--target agent` (default) | `repository-manager[agent]` — **agent runtime**, model orchestration + `epistemic-graph[full]` | `repository-manager-agent` |

```bash
docker build --target mcp   -t example/repository-manager:mcp    docker/   # connector-focused MCP server
docker build --target agent -t example/repository-manager:agent-local docker/   # agent runtime
```

`docker/mcp.compose.yml` runs the connector-focused `:mcp` server; `docker/agent.compose.yml` runs the
agent (`immutable agent digest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

Both `[mcp]` and `[agent]` carry the **epistemic-graph** engine through the required
Agent Utilities core dependency (`epistemic-graph[full]`). The `[mcp]` extra keeps
the server connector-focused; `[agent]` additionally enables model orchestration. Local
deployments can use the bundled engine. For production or shared state, run
**epistemic-graph as a dedicated database service** and configure the runtime to use it.
Deployment recipes (single-node + Raft HA), connection configuration, and architecture
diagrams are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=example&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/example)
![GitHub User's stars](https://img.shields.io/github/stars/example)

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/repository-manager/) and
is the recommended reference for installation, deployment, and day-to-day operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/repository-manager/installation/) | pip, source, extras, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/repository-manager/deployment/) | run the MCP and agent servers, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/repository-manager/usage/) | the MCP tools, the `Git` client, the CLI |
| [Overview](https://knuckles-team.github.io/repository-manager/overview/) | ecosystem role, enterprise readiness, architecture |
| [Phased Maintenance](https://knuckles-team.github.io/repository-manager/phased_maintenance/) | dependency-ordered workspace updates |
| [Phased Push](https://knuckles-team.github.io/repository-manager/phased_push/) | parallel, phase-gated multi-repository releases |
| [Concepts](https://knuckles-team.github.io/repository-manager/concepts/) | concept registry (`CONCEPT:RM-*`) |

`AGENTS.md` is the canonical contributor/agent guidance.

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-utilities-deployment (generated; do not edit between markers) -->

## Deploy with `agent-utilities-deployment`

Provision this package with the consolidated **`agent-utilities-deployment`**
workflow. It selects an installed-package, editable-source, or immutable-container
path; records only runtime secret and TLS-profile references in `AgentConfig`; and
runs doctor, registration, policy, observability, and rollback gates. Ask your agent
to **"deploy `repository-manager` with agent-utilities-deployment"**.

| Install mode | Command |
|------|---------|
| Installed package | `uv tool install "repository-manager[mcp]"`, then run `repository-manager-mcp` |
| Editable source | `uv pip install -e ".[agent]"`, then run `repository-manager-mcp` |
| Immutable container | deploy `registry.example.invalid/repository-manager@sha256:<digest>` through the operator-selected orchestrator |

The repository embeds no deployment profile, credential value, certificate path, or
environment-specific endpoint. Supply those at runtime through `AgentConfig` and the
configured secret provider.

<!-- END agent-utilities-deployment -->

<!-- GOVERNED-CAPABILITY:START -->
## Governed capability contract

This package ships a compact canonical skill surface with specialist procedures
kept as referenced workflows. The current MCP tools, skill metadata,
`connector_manifest.yml`, ontology, mappings, shapes, fixtures, migrations,
tool-schema fingerprints, and certification metadata form one versioned
capability contract. Validate them together; do not rely on stale tool names or
historical per-task skill wrappers.

Runtime endpoints, credentials, certificate trust, tenant identity, retention,
and observability policy are deployment inputs and are never packaged values.
See [Configuration, trust, and privacy](docs/configuration.md) before enabling a
network transport, connector ingestion, GraphOS delegation, or trace export.
<!-- GOVERNED-CAPABILITY:END -->
