# Installation

`repository-manager` is a standard Python package and a prebuilt container image.
Pick the path that matches how the operator want to run it.

## Requirements

- **Python 3.11+**.
- A reachable **Git** executable on `PATH` (the package shells out to `git` for bulk
  operations).
- A workspace directory containing the Git repositories the operator intend to manage — set
  via `REPOSITORY_MANAGER_WORKSPACE` (see [Deployment](deployment.md#configuration-environment)).

## From PyPI (recommended)

```bash
pip install repository-manager
```

### Optional extras

The base install is intentionally minimal. Install the extra for what the operator need:

| Extra | Install | Pulls in |
|---|---|---|
| `mcp` | `pip install "repository-manager[mcp]"` | FastMCP MCP-server runtime (`agent-utilities[mcp]`) |
| `agent` | `pip install "repository-manager[agent]"` | Pydantic-AI agent + Logfire tracing, `pre-commit`, `bump2version` |
| `test` | `pip install "repository-manager[test]"` | `pytest` tooling plus `agent-utilities[graphos,mcp]` and the certified `epistemic-graph` test kernel |
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

## Pre-commit in an isolated worktree

The framework-owned checks run from this repository's locked uv project, with
the live `agent-utilities` checkout linked through the ignored
`.uv-workspace-siblings/` path declared in `pyproject.toml`. The pytest hook
selects the repository's `test` extra, which includes the certified
`epistemic-graph` kernel required by RM's ingestion tests; it always collects
the current repository's test root, never a sibling checkout's `tests/` tree.
In a normal workspace the hook discovers the sibling automatically. For a
standalone clone, set `AGENT_UTILITIES_ROOT` to that checkout before running
`pre-commit run --all-files`.

Docker Compose validation uses the checked-in synthetic digest fixture at
`scripts/fixtures/precommit-compose.env`; it is never used for deployment.
Deployment still requires its own `REPOSITORY_MANAGER_MCP_IMAGE` and
`REPOSITORY_MANAGER_AGENT_IMAGE` values.

## Prebuilt Docker image

A multi-stage runtime image is published on every release (installs
`repository-manager[all]`):

```bash
docker pull example/repository-manager@sha256:<digest>

docker run --rm -i \
  -e REPOSITORY_MANAGER_WORKSPACE=/workspace \
  -v "${REPOSITORY_MANAGER_WORKSPACE}:/workspace" \
  example/repository-manager@sha256:<digest>        # stdio transport (default)
```

For an HTTP server with a published port, the agent server, and Docker Compose, see
[Deployment](deployment.md).

## Check the install

```bash
repository-manager --version
repository-manager-mcp --help
python -c "import repository_manager; print('repository-manager ready')"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP / agent server behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the `Git` client, and the CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.
