# Phased Git Push Workflow

The Phased Git Push feature orchestrates updating multiple repositories in sequential phases, utilizing parallelism within each phase and structured wait times between phases. This ensures dependencies can be built, published, and cached cleanly without causing race conditions across the ecosystem.

## Architecture

The phased push logic is configured within the `workspace.yml` under the `maintenance` section:

```yaml
maintenance:
  description: "Phased update sequence for the agent ecosystem."
  phases:
    - name: "Phase 1: Core Tools and UIs"
      phase: 1
      projects:
        - "universal-skills"
        - "skill-graphs"
        - "agent-webui"
        - "agent-terminal-ui"
      wait_minutes: 10
    - name: "Phase 2: agent-utilities"
      phase: 2
      projects:
        - "agent-utilities"
      wait_minutes: 5
    - name: "Phase 3: Agents"
      phase: 3
      bulk_push: True
```

### Key Capabilities
- **Parallel Push**: All projects defined in the same `phase` block are pushed concurrently using a ThreadPoolExecutor (`git push --follow-tags`).
- **Wait Durations**: The `wait_minutes` key introduces an active wait period (in minutes) upon completing a phase, preventing subsequent phases from building against stale/unpublished remote artifacts.
- **Bulk Execution**: A phase marked with `bulk_push: True` will dynamically resolve all repositories inside the workspace mapping that haven't been pushed in earlier phases (typically all the `agents/*` directories).

## Usage

### Command Line
You can trigger a phased push independently or as part of the maintenance lifecycle.

```bash
# Execute only the phased push sequence
repository-manager --push

# Execute a phased version bump, run pre-commit validations, then phased push
repository-manager --maintain --push

# Execute a single phase (e.g. Phase 2) and exit without continuing to Phase 3
repository-manager --push --phase 2 --single-phase

# Push only a specific project defined in the configuration
repository-manager --push --project agent-utilities
```

### MCP Tool Usage
The autonomous harness can trigger pushes via the MCP `phased_git_push` tool:

```json
{
  "name": "phased_git_push",
  "arguments": {
    "phase": 1,
    "target_project": null
  }
}
```

The server automatically handles progress reporting back to the Model Context Protocol UI as it progresses through the wait timers.
