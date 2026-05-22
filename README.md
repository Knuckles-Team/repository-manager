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

*Version: 1.18.0*

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

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](file:///home/apps/workspace/agent-packages/agents/repository-manager/docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools
| Tool Module | Toggle Env Var | Enabled by Default | Description & Nested Methods |
|-------------|----------------|--------------------|------------------------------|
| **Misc** | `MISCTOOL` | `True` | Register miscellaneous tools like health check. |
| **Git Operations** | `GIT_OPERATIONSTOOL` | `True` | Bulk Git operations and arbitrary command execution. Action-routed methods: `raw`, `clone`, `pull`, `push`, `phased_push`. |
| **Workspace Management** | `WORKSPACE_MANAGEMENTTOOL` | `True` | Register tools for core workspace setup and organization. Action-routed methods: `list`, `list_branches`, `setup`, `template`, `save`, `maintain`, `remediate`. |
| **Project Management** | `PROJECT_MANAGEMENT_TOOL` | `True` | Register tools for the autonomous project harness. Action-routed methods: `install`, `build`, `validate`, `validate_status`. |

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](file:///home/apps/workspace/agent-packages/agents/repository-manager/docs/mcp.md).

### MCP Configuration Examples

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "repository-manager": {
      "command": "uvx",
      "args": [
        "--from",
        "repository-manager",
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
        "repository-manager",
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
  knucklessg1/repository-manager:latest
```

---

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
    image: knucklessg1/repository-manager:latest
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

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](file:///home/apps/workspace/agent-packages/agents/repository-manager/docs/agent.md).

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

Install the Python package locally:

```bash
# Using uv (highly recommended)
uv pip install repository-manager[all]

# Using standard pip
python -m pip install repository-manager[all]
```

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`
