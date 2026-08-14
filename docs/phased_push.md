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
- **Change-aware start (`auto_start`, default on)**: By default the push begins at the *lowest phase that actually has unpushed work* instead of always Phase 1. Because phases are topologically ordered (lower phase = more upstream), a change in phase *N* can only cascade to phases `>= N` — so earlier, unchanged phases (and their `wait_minutes` pauses) are safely skipped. A repo counts as having work when it is not both clean and in sync with origin (uncommitted changes, an unpushed feature commit, or an unpushed version bump). If no repo has pending work, the push is a no-op. So editing a single Phase-2 repo triggers Phase 2 onward without sitting through the Phase-1 wait. Pass `auto_start=False` (CLI `--no-auto-start`) to opt out and start at `start_phase`; it also stands down automatically when a `target_project` / `project_filter` is set.

> **Explicit start as a floor.** `auto_start` only ever advances the start phase forward; an explicit `start_phase` (CLI `--phase`) still acts as a floor, and the two compose as `max(explicit, detected)`.

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
