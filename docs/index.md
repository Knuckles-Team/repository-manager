# repository-manager

Manage your git projects across a workspace — a **CLI, API, MCP server, and A2A
agent** for bulk Git operations, phased multi-repository releases, and
cross-repository graph intelligence in the agent-utilities ecosystem.

!!! info "Official documentation"
    This site is the canonical reference for `repository-manager`, maintained
    alongside every release.

[![PyPI](https://img.shields.io/pypi/v/repository-manager)](https://pypi.org/project/repository-manager/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/repository-manager)](https://github.com/Knuckles-Team/repository-manager/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/repository-manager)

## Overview

`repository-manager` operates a whole workspace of Git repositories as one unit. It
wraps bulk Git operations — clone, pull, push — with typed, deterministic interfaces,
and adds an autonomous release harness for **phased**, dependency-ordered version
bumps and pushes across the ecosystem. It provides:

- **`Git`** — a workspace-aware client (`repository_manager.repository_manager`) for
  cloning, pulling, branch enumeration, validation, and phased maintenance.
- **Action-routed MCP tools** — consolidated, togglable tool modules
  (`git_operations`, `workspace_management`, `project_management`, `misc`) that keep
  the LLM context small while exposing the full surface.
- **A cross-repository graph engine** — NetworkX in-memory analytics with LadybugDB
  Cypher persistence for impact, path, and dependency queries across the workspace.
- **An integrated Pydantic-AI graph agent** — orchestrating specialists over the
  Agent Control Protocol (ACP) and the Agent Web UI (AG-UI).

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, extras, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP and agent servers, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `Git` Python client, and the CLI.
- :material-sitemap: **[Overview](overview.md)** — ecosystem role, enterprise readiness, and architecture.
- :material-source-branch: **[Phased Maintenance](phased_maintenance.md)** — dependency-ordered workspace updates.
- :material-source-pull: **[Phased Push](phased_push.md)** — parallel, phase-gated multi-repository releases.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:RM-*` registry.

</div>

## Quick start

```bash
pip install "repository-manager[mcp]"
repository-manager-mcp            # stdio MCP server (default transport)
```

Point it at a workspace of Git repositories:

```bash
export REPOSITORY_MANAGER_WORKSPACE=/home/apps/workspace
repository-manager-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI extras, Docker image, all transports, the agent server, reverse
proxy, DNS).
