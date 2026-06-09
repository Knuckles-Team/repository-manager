# Installation

`repository-manager` is a standard Python package and a prebuilt container image.
Pick the path that matches how you want to run it.

## Requirements

- **Python 3.11+**.
- A reachable **Git** executable on `PATH` (the package shells out to `git` for bulk
  operations).
- A workspace directory containing the Git repositories you intend to manage — set
  via `REPOSITORY_MANAGER_WORKSPACE` (see [Deployment](deployment.md#configuration-environment)).

## From PyPI (recommended)

```bash
pip install repository-manager
```

### Optional extras

The base install is intentionally minimal. Install the extra for what you need:

| Extra | Install | Pulls in |
|---|---|---|
| `mcp` | `pip install "repository-manager[mcp]"` | FastMCP MCP-server runtime (`agent-utilities[mcp]`) |
| `agent` | `pip install "repository-manager[agent]"` | Pydantic-AI agent + Logfire tracing, `pre-commit`, `bump2version` |
| `test` | `pip install "repository-manager[test]"` | `pytest`, `pytest-xdist`, `pytest-asyncio`, `pytest-cov`, `pytest-timeout` |
| `all` | `pip install "repository-manager[all]"` | Everything above |

```bash
# Typical: run the MCP server and the release/maintenance harness
pip install "repository-manager[all]"
```

## From source

```bash
git clone https://github.com/Knuckles-Team/repository-manager.git
cd repository-manager
pip install -e ".[all]"          # editable install with every extra
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[all]"
uv run repository-manager-mcp
```

## Prebuilt Docker image

A multi-stage, slim image is published on every release (installs
`repository-manager[all]`):

```bash
docker pull knucklessg1/repository-manager:latest

docker run --rm -i \
  -e REPOSITORY_MANAGER_WORKSPACE=/workspace \
  -v /home/apps/workspace:/workspace \
  knucklessg1/repository-manager:latest        # stdio transport (default)
```

For an HTTP server with a published port, the agent server, and Docker Compose, see
[Deployment](deployment.md).

## Verify the install

```bash
repository-manager --version
repository-manager-mcp --help
python -c "import repository_manager; print('repository-manager ready')"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP / agent server behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the `Git` client, and the CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.
