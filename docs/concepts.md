# Concept Registry — repository-manager

> **Prefix**: `CONCEPT:RM-*`
> **Version**: 1.18.0
> **Bridge**: [`CONCEPT:MERISTEM-ECO.messaging.native-backend-abstraction`](https://github.com/Knuckles-Team/scion-sdk/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:RM-OS.governance.rm` | Git Operations | MCP tool domain `git_operations` — Action-routed dynamic tool registration |
| `CONCEPT:RM-OS.governance.rm-2` | Misc Operations | MCP tool domain `misc` — Action-routed dynamic tool registration |
| `CONCEPT:RM-OS.governance.rm-3` | Project Management | MCP tool domain `project_management` — Action-routed dynamic tool registration |
| `CONCEPT:RM-OS.governance.rm-4` | Workspace Management | MCP tool domain `workspace_management` — Action-routed dynamic tool registration |
| `CONCEPT:RM-SAFE-COMMIT` | Safe commit | Stages the complete tree before a gate and proves `nothing_left_unstaged` before commit (C-12) |
| `CONCEPT:RM-DESTRUCTIVE-GUARD` | Destructive git guard | Refuses destructive verbs unless a logged, single-use, snapshot-backed override is supplied (C-12) |
| `CONCEPT:RM-TREE-REPAIR` | Working-tree repair | Detects index collapse and `core.bare` drift and applies content-preserving mechanical repairs (C-12) |

## Cross-Project References (from scion-sdk)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:MERISTEM-ECO.messaging.native-backend-abstraction` | Unified Toolkit Ingestion | scion-sdk |
| `CONCEPT:MERISTEM-ORCH.adapter.hot-cache-invalidation` | Confidence-Gated Router | scion-sdk |
| `CONCEPT:MERISTEM-OS.config.secrets-authentication` | Prompt Injection Defense | scion-sdk |
| `CONCEPT:MERISTEM-OS.state.cognitive-scheduler-preemption` | Cognitive Scheduler | scion-sdk |
| `CONCEPT:MERISTEM-OS.governance.reactive-multi-axis-budget` | Guardrail Engine | scion-sdk |
| `CONCEPT:MERISTEM-OS.governance.wasm-micro-agent-sandbox` | Audit Logging | scion-sdk |
| `CONCEPT:MERISTEM-KG.query.object-graph-mapper` | Knowledge Graph Core | scion-sdk |

## Synergy with scion-sdk

This project integrates with `scion-sdk` via `CONCEPT:MERISTEM-ECO.messaging.native-backend-abstraction` (Unified Toolkit Ingestion). The `repository_manager` MCP server registers its tools with the scion-sdk FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all RM-* concepts.
