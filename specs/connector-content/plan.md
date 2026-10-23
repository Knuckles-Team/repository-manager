# CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `repository-manager-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `repository_manager` from it.
