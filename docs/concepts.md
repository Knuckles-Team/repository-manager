# Concept Registry — repository-manager

> **Prefix**: `CONCEPT:RM-*`
> **Version**: 1.18.0
> **Bridge**: [`CONCEPT:ECO-4.0`](../../agent-utilities/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:RM-001` | Git Operations | MCP tool domain `git_operations` — Action-routed dynamic tool registration |
| `CONCEPT:RM-002` | Misc Operations | MCP tool domain `misc` — Action-routed dynamic tool registration |
| `CONCEPT:RM-003` | Project Management | MCP tool domain `project_management` — Action-routed dynamic tool registration |
| `CONCEPT:RM-004` | Workspace Management | MCP tool domain `workspace_management` — Action-routed dynamic tool registration |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:ECO-4.0` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:ORCH-1.2` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:OS-5.1` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:OS-5.2` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:OS-5.3` | Guardrail Engine | agent-utilities |
| `CONCEPT:OS-5.4` | Audit Logging | agent-utilities |
| `CONCEPT:KG-2.0` | Knowledge Graph Core | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:ECO-4.0` (Unified Toolkit Ingestion). The `repository_manager` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all RM-* concepts.
