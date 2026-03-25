# Agent Packages Documentation

This directory contains a collection of agent repositories that generally follow a standard architecture. These projects serve as MCP (Model Context Protocol) servers, A2A (Agent-to-Agent) endpoints, and AG-UI (Agent UI) backends.

## Overview

Each repository in this directory typically exposes:

1.  **MCP Server**: Implements tools and resources compatible with the Model Context Protocol.
2.  **A2A Endpoint**: Allows other agents to discover and interact with this agent via the `fastA2A` protocol.
3.  **AG-UI Endpoint**: Provides a streaming UI compatible with the Agent UI standard.
4.  **Web UI**: A simple web interface for direct user interaction (optional).

## Architecture

The architecture consists of three main layers:

*   **Base Layer (MCP)**: Defined in `*mcp_server.py`. This layer uses `FastMCP` to define tools and resources. It handles the core logic of the tools.
*   **Agent Layer (Pydantic-AI)**: Defined in `*agent_server.py`. This layer uses `pydantic-ai` to create an `Agent` instance. It binds the MCP tools to the agent and defines the system prompt and orchestration logic.
*   **Interface Layer (FastAPI)**: Defined in `*agent_server.py` or `server.py`. This layer uses `FastAPI` and `uvicorn` to serve the agent over HTTP/SSE. It exposes the `/mcp`, `/a2a`, and `/ag-ui` endpoints.

## Implementation Patterns

The ecosystem supports both Flat Agents for direct tool usage and Graph Agents for complex supervised workflows.

## Agent Catalog

| Agent Package | Type |
|:--------------|:-----|
<!-- AGENT_CATALOG_PLACEHOLDER -->

---

## New Repository Skeleton Guide

Follow these standards when creating a new agent repository. Use the `agent-package-builder` skill from `universal-skills` for automated scaffolding.

### 1. Standard File Structure

```text
my-new-agent/
├── pyproject.toml          # Python package config
├── requirements.txt        # Dependencies
├── Dockerfile              # Production build
├── debug.Dockerfile        # Dev build with reloading
├── compose.yaml            # Docker Compose service definition
├── .pre-commit-config.yaml # Linting (ruff, black, etc.)
├── .bumpversion.cfg        # Version management
├── .env                    # Environment variables (template)
├── my_new_agent/           # Source code
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent_server.py     # Agent and Server definition
│   ├── mcp_server.py       # Tool definitions (FastMCP)
│   └── mcp_config.json     # Connections to other MCP servers (optional)
└── scripts/                # Helper scripts
```

---

## Core Infrastructure

This workspace relies on a foundation of shared utilities and specialized tools:

- **[`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities.git)**: Common code for servers, agents, and base utilities.
- **[`universal-skills`](https://github.com/Knuckles-Team/universal-skills.git)**: Centralized library of agent capabilities and skills.
- **[`agent-webui`](https://github.com/Knuckles-Team/agent-webui.git)**: Standardized UI components for agent interfaces.
- **[`skill-graphs`](https://github.com/Knuckles-Team/skill-graphs.git)**: Management of complex skill interactions and graph definitions.
