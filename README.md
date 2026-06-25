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

*Version: 1.38.0*

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

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `rm_git` | `GIT_OPERATIONSTOOL` | Bulk Git operations and arbitrary command execution. |
| `rm_projects` | `PROJECT_MANAGEMENTTOOL` | Bulk install, build, and validate Python projects. |
| `rm_workspace` | `WORKSPACE_MANAGEMENTTOOL` | Core workspace organization, configuration, and maintenance. |
| `rm_worktree` | `PROJECT_MANAGEMENTTOOL` | Manage git worktrees for concurrent multi-session development (CONCEPT:RM-WORKTREE). |

_4 action-routed tools (default `MCP_TOOL_MODE=condensed`). Each is enabled unless its toggle is set false; set `MCP_TOOL_MODE=verbose` (or `both`) for the 1:1 per-operation surface. Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

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

> **Install the slim `[mcp]` extra.** All examples below install
> `repository-manager[mcp]` — the MCP-server extra that pulls only the FastMCP /
> FastAPI tooling (`agent-utilities[mcp]`). It deliberately **excludes** the heavy
> agent runtime (the epistemic-graph engine, `pydantic-ai`, `dspy`, `llama-index`,
> `tree-sitter`), so `uvx`/container installs are dramatically smaller and faster.
> Use the full `[agent]` extra only when you need the integrated Pydantic AI agent
> (see [Installation](#installation)).

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "repository-manager": {
      "command": "uvx",
      "args": [
        "--from",
        "repository-manager[mcp]",
        "repository-manager-mcp"
      ],
      "env": {
        "REPO_MANAGER_URL": "your_repo_manager_url_here",
        "REPO_MANAGER_USERNAME": "your_repo_manager_username_here",
        "REPOSITORY_MANAGER_WORKSPACE": "your_repository_manager_workspace_here",
        "LLM_ROUTER_MODEL": "your_llm_router_model_here",
        "LLM_AGENT_MODEL": "your_llm_agent_model_here",
        "GRAPH_ROUTER_TIMEOUT": "your_graph_router_timeout_here",
        "GRAPH_VERIFIER_TIMEOUT": "your_graph_verifier_timeout_here",
        "REPO_MANAGER_PASSWORD": "your_repo_manager_password_here"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (Recommended for production deployments)
Configure your client's `mcp.json` to launch the Streamable-HTTP server via `uvx` with explicit host and port definition:

```json
{
  "mcpServers": {
    "repository-manager": {
      "command": "uvx",
      "args": [
        "--from",
        "repository-manager[mcp]",
        "repository-manager-mcp"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "REPO_MANAGER_URL": "your_repo_manager_url_here",
        "REPO_MANAGER_USERNAME": "your_repo_manager_username_here",
        "REPOSITORY_MANAGER_WORKSPACE": "your_repository_manager_workspace_here",
        "LLM_ROUTER_MODEL": "your_llm_router_model_here",
        "LLM_AGENT_MODEL": "your_llm_agent_model_here",
        "GRAPH_ROUTER_TIMEOUT": "your_graph_router_timeout_here",
        "GRAPH_VERIFIER_TIMEOUT": "your_graph_verifier_timeout_here",
        "REPO_MANAGER_PASSWORD": "your_repo_manager_password_here"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed remote or local Streamable-HTTP instance:

```json
{
  "mcpServers": {
    "repository-manager": {
      "url": "http://localhost:8000/repository-manager/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name repository-manager-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e REPO_MANAGER_URL="your_value" \
  -e REPO_MANAGER_USERNAME="your_value" \
  -e REPOSITORY_MANAGER_WORKSPACE="your_value" \
  -e LLM_ROUTER_MODEL="your_value" \
  -e LLM_AGENT_MODEL="your_value" \
  -e GRAPH_ROUTER_TIMEOUT="your_value" \
  -e GRAPH_VERIFIER_TIMEOUT="your_value" \
  -e REPO_MANAGER_PASSWORD="your_value" \
  knucklessg1/repository-manager:mcp
```

> The `:mcp` tag is the **slim MCP-server image** (built from
> `docker/Dockerfile --target mcp`, installing `repository-manager[mcp]`). The default
> `:latest` tag is the **full agent image** (`--target agent`, `repository-manager[agent]`)
> which also bundles the Pydantic AI agent and the epistemic-graph engine — use it
> when you run `repository-manager-agent` (the agent), not just the MCP server. See
> [Container images](#container-images-mcp-vs-agent).

---

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`repository-manager` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/repository-manager/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://repository-manager-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set credentials
export REPO_MANAGER_URL="your_value"
export REPO_MANAGER_USERNAME="your_value"
export REPOSITORY_MANAGER_WORKSPACE="your_value"
export LLM_ROUTER_MODEL="your_value"
export LLM_AGENT_MODEL="your_value"
export GRAPH_ROUTER_TIMEOUT="your_value"
export GRAPH_VERIFIER_TIMEOUT="your_value"
export REPO_MANAGER_PASSWORD="your_value"

# Run the agent server
repository-manager-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:mcp
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
    image: knucklessg1/repository-manager:latest
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

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](docs/agent.md).

---

## Environment Variables

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
| `REPOSITORY_MANAGER_WORKSPACE` | Root directory of the git workspace to manage | — |
| `REPOSITORY_MANAGER_WORKTREE_ROOT` | Root directory for managed git worktrees | — |
| `REPOSITORY_MANAGER_DEFAULT_BRANCH` | Default branch name for git operations | — |
| `REPOSITORY_MANAGER_THREADS` | Parallel threads for git operations | — |
| `REPO_MANAGER_URL` | Base URL of the target service | `http://localhost:8000` |
| `REPO_MANAGER_USERNAME` | Username for authentication | `admin` |
| `REPO_MANAGER_PASSWORD` | Password for authentication | — |

### Hybrid graph intelligence engine
| Variable | Description | Default |
|----------|-------------|---------|
| `GRAPH_DB_PATH` | LadybugDB persistent graph store path | — |
| `GRAPH_SYNC_BACKGROUND` | Run graph sync in the background | — |
| `GRAPH_ROUTER_TIMEOUT` | Router-node timeout (seconds) | `300.0` |
| `GRAPH_VERIFIER_TIMEOUT` | Verifier-node timeout (seconds) | `300.0` |
| `LLM_ROUTER_MODEL` | Model used by the router node | — |
| `LLM_AGENT_MODEL` | Model used by specialist agent nodes | — |

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
| `repository-manager[mcp]` | Slim MCP server only (`agent-utilities[mcp]` — FastMCP/FastAPI) | You only run the **MCP server** (smallest install / image) |
| `repository-manager[agent]` | Full agent runtime (`agent-utilities[agent,logfire]` — Pydantic AI + the epistemic-graph engine, plus `pre-commit` / `bump2version`) | You run the **integrated agent** |
| `repository-manager[all]` | Everything (`mcp` + `agent` + `logfire`) | Development / both surfaces |

```bash
# MCP server only (recommended for tool hosting — slim deps)
uv pip install "repository-manager[mcp]"

# Full agent runtime (Pydantic AI + epistemic-graph engine)
uv pip install "repository-manager[agent]"

# Everything (development)
uv pip install "repository-manager[all]"      # or: python -m pip install "repository-manager[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `knucklessg1/repository-manager:mcp` | `--target mcp` | `repository-manager[mcp]` — **slim**, no engine/`pydantic-ai`/`dspy`/`llama-index`/`tree-sitter` | `repository-manager-mcp` |
| `knucklessg1/repository-manager:latest` | `--target agent` (default) | `repository-manager[agent]` — **full** agent runtime + epistemic-graph engine | `repository-manager-agent` |

```bash
docker build --target mcp   -t knucklessg1/repository-manager:mcp    docker/   # slim MCP server
docker build --target agent -t knucklessg1/repository-manager:latest docker/   # full agent
```

`docker/mcp.compose.yml` runs the slim `:mcp` server; `docker/agent.compose.yml` runs the
agent (`:latest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

The **full agent** (`[agent]` / `:latest`) embeds the **epistemic-graph** engine (pulled in
transitively via `agent-utilities[agent]`). For production — or to share one knowledge graph
across multiple agents — run **epistemic-graph as its own database container** and point the
agent at it instead of embedding it. Deployment recipes (single-node + Raft HA), connection
config, and the full database architecture (with diagrams) are documented in the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).
The slim `[mcp]` server does **not** require the database.

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

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


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `repository-manager` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx repository-manager-mcp` · or `uv tool install repository-manager` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/repository-manager:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->
