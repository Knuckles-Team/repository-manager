# Deployment

This page covers running `repository-manager` as a long-lived server: the
transports, a Docker Compose stack, the companion agent server, putting it behind a
Caddy reverse proxy, and giving it a DNS name with Technitium.

> `repository-manager` ships **both** an MCP server (console script
> `repository-manager-mcp`) and an A2A agent server (console script
> `repository-manager-agent`). The MCP server is a typed, deterministic tool surface
> a policy router / agent calls; the agent server wraps a Pydantic-AI graph
> orchestrator exposed over ACP and the Agent Web UI.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    repository-manager-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    repository-manager-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    repository-manager-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`repository-manager` is configured entirely from the environment. The **required**
set:

| Var | Default | Meaning |
|---|---|---|
| `REPOSITORY_MANAGER_WORKSPACE` | `/home/apps/workspace` | Root directory containing the managed Git repositories |
| `REPO_MANAGER_URL` | `http://localhost:8000` | Base URL of the running service |
| `REPO_MANAGER_USERNAME` | `admin` | Service identity |
| `REPO_MANAGER_PASSWORD` | _(unset)_ | Service credential / token |
| `HOST` | `0.0.0.0` | Bind address (HTTP transports) |
| `PORT` | `8000` | Bind port (HTTP transports) |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |
| `MISCTOOL` | `True` | Register the miscellaneous tool set |
| `GIT_OPERATIONSTOOL` | `True` | Register the bulk Git tool set |
| `WORKSPACE_MANAGEMENTTOOL` | `True` | Register the workspace-management tool set |
| `PROJECT_MANAGEMENT_TOOL` | `True` | Register the project-management tool set |

The graph agent additionally reads `LLM_ROUTER_MODEL`, `LLM_AGENT_MODEL`,
`GRAPH_ROUTER_TIMEOUT`, and `GRAPH_VERIFIER_TIMEOUT`. Telemetry (`ENABLE_OTEL`,
`OTEL_EXPORTER_OTLP_*`) and access governance (`EUNOMIA_TYPE`,
`EUNOMIA_POLICY_FILE`, `EUNOMIA_REMOTE_URL`) are optional. The full set, with
required-vs-optional separation, is documented in
[`.env.example`](https://github.com/Knuckles-Team/repository-manager/blob/main/.env.example).
Copy it to `.env` and populate only what you use.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/repository-manager/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
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
```

```bash
cp .env.example .env          # then edit REPOSITORY_MANAGER_WORKSPACE etc.
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

## Agent server

`repository-manager` ships a second console script, `repository-manager-agent`, that
runs the integrated Pydantic-AI **graph agent** over the Agent Control Protocol
(ACP) and the Agent Web UI (AG-UI). It connects to the MCP server it orchestrates via
`MCP_URL` and listens on port `9047` by default.

```bash
export MCP_URL=http://localhost:8000/mcp
export PROVIDER=openai
export MODEL_ID=gpt-4o
repository-manager-agent
```

The repo ships [`docker/agent.compose.yml`](https://github.com/Knuckles-Team/repository-manager/blob/main/docker/agent.compose.yml),
which provisions the MCP server and the agent server together — the agent reaches the
MCP server by container name:

```yaml
services:
  repository-manager-mcp:
    image: knucklessg1/repository-manager:latest
    hostname: repository-manager-mcp
    env_file: [../.env]
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports: ["8000:8000"]

  repository-manager-agent:
    image: knucklessg1/repository-manager:latest
    depends_on: [repository-manager-mcp]
    command: ["repository-manager-agent"]
    env_file: [../.env]
    environment:
      - HOST=0.0.0.0
      - PORT=9047
      - MCP_URL=http://repository-manager-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports: ["9047:9047"]
```

```bash
docker compose -f docker/agent.compose.yml up -d
```

## Behind a Caddy reverse proxy

Expose the HTTP server on a hostname with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
repository-manager.arpa {
    tls internal
    reverse_proxy repository-manager-mcp:8000
}
```

```caddy
# Public — automatic Let's Encrypt
repository-manager.example.com {
    reverse_proxy repository-manager-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=repository-manager.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `repository-manager.arpa → <caddy-host-ip>` in the Technitium
web console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json` (multiplexer nickname `rep`):

```json
{
  "mcpServers": {
    "repository-manager": {
      "command": "uv",
      "args": ["run", "repository-manager-mcp"],
      "env": {
        "REPOSITORY_MANAGER_WORKSPACE": "/home/apps/workspace"
      }
    }
  }
}
```

For a remote HTTP server, point the client at
`http://repository-manager.arpa/mcp` instead.
