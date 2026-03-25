# Repository Manager - A2A | AG-UI | MCP

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

*Version: 1.3.46*

## Overview

- **Pydantic Graph Architecture**: 19 specialized domain nodes (Git, File, Workspace, and 15+ integrated engineering skills) for intelligent, granular task routing.
- **Declarative Workspace**: Manage your entire ecosystem via `workspace.yml`, validated by strict Pydantic V2 models.
- **Idempotent Synchronization**: One-click setup that intelligently clones missing repositories and pulls existing ones into their correct hierarchical paths.
- **Workspace Visualization**:
    - **ASCII Tree**: Generate beautiful folder structures directly in the CLI or via MCP.
    - **Mermaid Diagrams**: Export your workspace model as a visual graph for documentation.
- **Integrated Skills**: Native support for `agent-builder`, `mcp-builder`, `web-search`, and more, coupled with expert documentation for `FastMCP`, `Pydantic AI`, and `Docker`.

1. **Nodes**: Specialized agents for high-context domains (e.g., `GitOpsNode`, `KnowledgeNode`).
2. **Router**: Automatically directs intent based on tool tags (`git_operations`, `workspace_management`, etc.).
3. **Engine**: The core `WorkspaceManager` processes the `workspace.yml` model to maintain state.

## 🛠️ Usage

### Workspace Configuration (`workspace.yml`)
Define your world in a single file:

```yaml
name: "My Workspace"
path: "./workspace"
repositories:
  - url: "https://github.com/org/repo-core.git"
subdirectories:
  agents:
    repositories:
      - url: "https://github.com/org/agent-1.git"
maintenance:
  phases:
    - name: "Phase 1: Core"
      phase: 1
      project: "repo-core"
```

## MCP

AI Prompt:
```text
Setup my workspace using the workspace.yml configuration. Also, install and validate all projects in the workspace.
```

AI Response:
```text
Workspace setup complete: Missing repositories have been cloned and existing ones updated.
Bulk operations finished: All projects installed and validated (agent/mcp) across the workspace.
```

This repository is actively maintained - Contributions are welcome!

## A2A Agent

### Architecture:

```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph subGraph0["Agent Capabilities"]
        C["Agent"]
        B["A2A Server - Uvicorn/FastAPI"]
        D["MCP Tools"]
        F["Agent Skills"]
  end
    C --> D & F
    A["User Query"] --> B
    B --> C
    D --> E["Platform API"]

     C:::agent
     B:::server
     A:::server
    classDef server fill:#f9f,stroke:#333
    classDef agent fill:#bbf,stroke:#333,stroke-width:2px
    style B stroke:#000000,fill:#FFD600
    style D stroke:#000000,fill:#BBDEFB
    style F fill:#BBDEFB
    style A fill:#C8E6C9
    style subGraph0 fill:#FFF9C4
```

### Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant Server as A2A Server
    participant Agent as Agent
    participant Skill as Agent Skills
    participant MCP as MCP Tools

    User->>Server: Send Query
    Server->>Agent: Invoke Agent
    Agent->>Skill: Analyze Skills Available
    Skill->>Agent: Provide Guidance on Next Steps
    Agent->>MCP: Invoke Tool
    MCP-->>Agent: Tool Response Returned
    Agent-->>Agent: Return Results Summarized
    Agent-->>Server: Final Response
    Server-->>User: Output
```

## Usage

### CLI

| Short Flag | Long Flag        | Description                                  |
|------------|------------------|----------------------------------------------|
| -h         | --help           | See Usage                                    |
| -b         | --default-branch | Checkout default branch                      |
| -c         | --clone          | Clone projects specified in workspace file   |
| -p         | --pull           | Pull all projects in workspace               |
| -w         | --workspace      | Specify the workspace root directory         |
| -f         | --file           | Specify the workspace YAML file (Default)    |
| -r         | --repositories   | Comma separated Git URLs (Override)          |
| -t         | --threads        | Number of parallel threads (Default: 12)     |
| -m         | --maintain       | Run phased maintenance workflow              |
|            | --pre-commit     | Run parallel pre-commit checks               |
|            | --bump           | Bulk version bump (patch, minor, major)      |
|            | --phase          | Start maintenance at Phase N (1-5)           |
|            | --dry-run        | Preview changes without applying them        |
|            | --skip-pre-commit| Skip pre-commit phase in maintenance         |
|            | --install        | [NEW] Bulk install all Python projects       |
|            | --build          | [NEW] Bulk build all Python projects         |
|            | --validate       | [NEW] Bulk validate all agent/MCP servers    |
|            | --type           | Validation filter: agent, mcp, or all        |
|            | --tree           | [NEW] Generate ASCII workspace tree          |
|            | --mermaid        | [NEW] Generate Mermaid workspace diagram     |
|            | --setup          | [NEW] Sync workspace from YAML config        |

```bash
repository-manager \
    --clone  \
    --pull  \
    --workspace '/home/user/Downloads'  \
    --file '/home/user/Downloads/repositories.txt'  \
    --repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot' \
    --threads 8
```

### MCP CLI

| Short Flag | Long Flag                          | Description                                                                 |
|------------|------------------------------------|-----------------------------------------------------------------------------|
| -h         | --help                             | Display help information                                                    |
| -t         | --transport                        | Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)       |
| -s         | --host                             | Host address for HTTP transport (default: 0.0.0.0)                          |
| -p         | --port                             | Port number for HTTP transport (default: 8000)                              |
|            | --auth-type                        | Authentication type: 'none', 'static', 'jwt', 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (default: none) |
|            | --token-jwks-uri                   | JWKS URI for JWT verification                                              |
|            | --token-issuer                     | Issuer for JWT verification                                                |
|            | --token-audience                   | Audience for JWT verification                                              |
|            | --oauth-upstream-auth-endpoint     | Upstream authorization endpoint for OAuth Proxy                             |
|            | --oauth-upstream-token-endpoint    | Upstream token endpoint for OAuth Proxy                                    |
|            | --oauth-upstream-client-id         | Upstream client ID for OAuth Proxy                                         |
|            | --oauth-upstream-client-secret     | Upstream client secret for OAuth Proxy                                     |
|            | --oauth-base-url                   | Base URL for OAuth Proxy                                                   |
|            | --oidc-config-url                  | OIDC configuration URL                                                     |
|            | --oidc-client-id                   | OIDC client ID                                                             |
|            | --oidc-client-secret               | OIDC client secret                                                         |
|            | --oidc-base-url                    | Base URL for OIDC Proxy                                                    |
|            | --remote-auth-servers              | Comma-separated list of authorization servers for Remote OAuth             |
|            | --remote-base-url                  | Base URL for Remote OAuth                                                  |
|            | --allowed-client-redirect-uris     | Comma-separated list of allowed client redirect URIs                       |
|            | --eunomia-type                     | Eunomia authorization type: 'none', 'embedded', 'remote' (default: none)   |
|            | --eunomia-policy-file              | Policy file for embedded Eunomia (default: mcp_policies.json)              |
|            | --eunomia-remote-url               | URL for remote Eunomia server                                              |


### A2A CLI

| Short Flag | Long Flag         | Description                                                            |
|------------|-------------------|------------------------------------------------------------------------|
| -h         | --help            | Display help information                                               |
|            | --host            | Host to bind the server to (default: 0.0.0.0)                          |
|            | --port            | Port to bind the server to (default: 9000)                             |
|            | --reload          | Enable auto-reload                                                     |
|            | --provider        | LLM Provider: 'openai', 'anthropic', 'google', 'huggingface'           |
|            | --model-id        | LLM Model ID (default: qwen3:4b)                                       |
|            | --base-url        | LLM Base URL (for OpenAI compatible providers)                         |
|            | --api-key         | LLM API Key                                                            |
|            | --smart-coding-mcp-enable | Enable Smart Coding MCP configuration                                  |
|            | --python-sandbox-enable | Enable Python Sandbox MCP configuration                                  |
|            | --workspace            | Workspace to scan for git projects (default: current directory)       |


### Smart Coding MCP Integration

The Repository Manager A2A Agent can automatically configure `smart-coding-mcp` for any Git projects found in a specified directory.

```bash
repository_manager_a2a --smart-coding-mcp-enable --workspace /path/to/my/projects
```

This will:
1. Scan `/path/to/my/projects` for any subdirectories containing a `.git` folder.
2. Update `mcp_config.json` to include a `smart-coding-mcp` server entry for each found project.
3. Start the agent with access to these new MCP servers, allowing for semantic code search within your projects.

### Python Sandbox Integration

The Agent can execute Python code in a secure Deno sandbox using `mcp-run-python`.

```bash
repository_manager_a2a --python-sandbox-enable
```

This will:
1.  Configure `mcp_config.json` to include the `python-sandbox` server.
2.  Enable the `Python Sandbox` skill, allowing the agent to run scripts for calculation, testing, or logic verification.

### Default Workspace Model

The manager automatically discovers `workspace.yml` in the current directory or via the `WORKSPACE_YML` environment variable. This file serves as the strict single source of truth for the entire environment hierarchy, encompassing repositories, subdirectories, and maintenance policies.

### Maintenance Workflows

`repository-manager` supports specialized maintenance workflows for managing interdependent package ecosystems.

#### Parallel Pre-commits
Run `pre-commit` checks across all repositories in parallel. This is significantly faster than sequential runs and simplifies fleet-wide health checks.

```bash
repository-manager --pre-commit
```

#### Phased Bumping
When packages depend on each other, they often need to be bumped in a specific sequence. The `--maintain` flag implements this 5-stage process:

1.  **Skills**: Update core skill packages.
2.  **Graphs**: Update AI graph/template repositories.
3.  **UI**: Update frontend components.
4.  **Utilities**: Update the central utility library (`agent-utilities`) and propagate skill/graph versions.
5.  **Fleet**: Propagate the new utility version to all other packages and bump their versions.

```bash
# Full maintenance run
repository-manager --maintain

# Skip verify phase (pre-commit) if already done
repository-manager --maintain --skip-pre-commit

# Resume from a specific phase
repository-manager --maintain --phase 4
```


### Using as an MCP Server

The MCP Server can be run in two modes: `stdio` (for local testing) or `http` (for networked access). To start the server, use the following commands:

#### Run in stdio mode (default):
```bash
repository-manager-mcp --transport "stdio"
```

#### Run in HTTP mode:
```bash
repository-manager-mcp --transport "http"  --host "0.0.0.0"  --port "8000"
```

### Use in Python

```python
from repository_manager.repository_manager import Git

gitlab = Git()

gitlab.set_workspace("<workspace>")

gitlab.set_threads(threads=8)

gitlab.set_git_projects("<projects>")

gitlab.set_default_branch(set_to_default_branch=True)

gitlab.clone_projects_in_parallel()

gitlab.pull_projects_in_parallel()
```


### Deploy MCP Server as a Service

The ServiceNow MCP server can be deployed using Docker, with configurable authentication, middleware, and Eunomia authorization.

#### Using Docker Run

```bash
docker pull knucklessg1/repository-manager:latest

docker run -d \
  --name repository-manager-mcp \
  -p 8004:8004 \
  -e HOST=0.0.0.0 \
  -e PORT=8004 \
  -e TRANSPORT=http \
  -e AUTH_TYPE=none \
  -e EUNOMIA_TYPE=none \
  -v development:/root/Development \
  knucklessg1/repository-manager:latest
```

For advanced authentication (e.g., JWT, OAuth Proxy, OIDC Proxy, Remote OAuth) or Eunomia, add the relevant environment variables:

```bash
docker run -d \
  --name repository-manager-mcp \
  -p 8004:8004 \
  -e HOST=0.0.0.0 \
  -e PORT=8004 \
  -e TRANSPORT=http \
  -e AUTH_TYPE=oidc-proxy \
  -e OIDC_CONFIG_URL=https://provider.com/.well-known/openid-configuration \
  -e OIDC_CLIENT_ID=your-client-id \
  -e OIDC_CLIENT_SECRET=your-client-secret \
  -e OIDC_BASE_URL=https://your-server.com \
  -e ALLOWED_CLIENT_REDIRECT_URIS=http://localhost:*,https://*.example.com/* \
  -e EUNOMIA_TYPE=embedded \
  -e EUNOMIA_POLICY_FILE=/app/mcp_policies.json \
  -v development:/root/Development \
  knucklessg1/repository-manager:latest
```

#### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8004
      - TRANSPORT=http
      - AUTH_TYPE=none
      - EUNOMIA_TYPE=none
    volumes:
      - development:/root/Development
    ports:
      - 8004:8004
```

For advanced setups with authentication and Eunomia:

```yaml
services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8004
      - TRANSPORT=http
      - AUTH_TYPE=oidc-proxy
      - OIDC_CONFIG_URL=https://provider.com/.well-known/openid-configuration
      - OIDC_CLIENT_ID=your-client-id
      - OIDC_CLIENT_SECRET=your-client-secret
      - OIDC_BASE_URL=https://your-server.com
      - ALLOWED_CLIENT_REDIRECT_URIS=http://localhost:*,https://*.example.com/*
      - EUNOMIA_TYPE=embedded
      - EUNOMIA_POLICY_FILE=/app/mcp_policies.json
    ports:
      - 8004:8004
    volumes:
      - development:/root/Development
      - ./mcp_policies.json:/app/mcp_policies.json
```

Run the service:

```bash
docker-compose up -d
```

#### Configure `mcp.json` for AI Integration

```json
{
  "mcpServers": {
    "repository_manager": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "repository-manager",
        "repository-manager-mcp"
      ],
      "env": {
        "REPOSITORY_MANAGER_WORKSPACE": "/home/user/Development/",                       // Optional - Can be specified at prompt
        "REPOSITORY_MANAGER_THREADS": "12",                                              // Optional - Can be specified at prompt
        "REPOSITORY_MANAGER_DEFAULT_BRANCH": "True",                                     // Optional - Can be specified at prompt
        "REPOSITORY_MANAGER_PROJECTS_FILE": "/home/user/Development/repositories.txt"    // Optional - Can be specified at prompt
      },
      "timeout": 300000
    }
  }
}

```

### A2A

#
#### Endpoints
- **Web UI**: `http://localhost:8000/` (if enabled)
- **A2A**: `http://localhost:8000/a2a` (Discovery: `/a2a/.well-known/agent.json`)
- **AG-UI**: `http://localhost:8000/ag-ui` (POST)

#### A2A CLI

| Short Flag | Long Flag         | Description                                                            |
|------------|-------------------|------------------------------------------------------------------------|
| -h         | --help            | Display help information                                               |
|            | --host            | Host to bind the server to (default: 0.0.0.0)                          |
|            | --port            | Port to bind the server to (default: 9000)                             |
|            | --reload          | Enable auto-reload                                                     |
|            | --provider        | LLM Provider: 'openai', 'anthropic', 'google', 'huggingface'           |
|            | --model-id        | LLM Model ID (default: qwen3:4b)                                       |
|            | --base-url        | LLM Base URL (for OpenAI compatible providers)                         |
|            | --api-key         | LLM API Key                                                            |
|            | --api-key         | LLM API Key                                                            |
| --mcp-url         | MCP Server URL (default: http://localhost:8000/mcp)                    |
| --web             | Enable Pydantic AI Web UI                                              | False (Env: ENABLE_WEB_UI) |


## Install Python Package

```bash
pip install repository-manager
```

or

```bash
uv pip install --upgrade repository-manager
```


## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
