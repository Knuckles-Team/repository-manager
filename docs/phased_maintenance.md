# Phased Maintenance Workflows

The `repository-manager` includes powerful workflow automation features to execute phased, sequential updates across the workspace. This is primarily controlled via the `workspace.yml` configuration and the command-line interface.

## Phased Maintenance Phases

The agent ecosystem enforces dependency order updates through sequential maintenance phases. By default, three core phases govern the ecosystem:

1. **Phase 1: Core Tools and UIs**
   Updates the foundation (`universal-skills`, `skill-graphs`, `agent-webui`, `agent-terminal-ui`).
   Wait Interval: 10 minutes (allows CI/CD or CD pipeline deployments to propagate).

2. **Phase 2: agent-utilities**
   Updates the primary orchestrator that depends on core tools.
   Wait Interval: 5 minutes.

3. **Phase 3: Agents**
   Executes a bulk update for all `agent` projects using a wildcard strategy to deploy final updates to the ecosystem.

## CLI Usage

The following flags control phased update and push sequences:

- `--validate`: Executes a full pre-release validation. If this fails, the next steps are aborted.
- `--bump [patch/minor/major]`: Executes a version bump.
- `--maintain`: Executes phased dependency updates across the workspace. Modifies `pyproject.toml` automatically based on dependency tree.
- `--push`: Executes a parallelized Git Push sequence per-phase, respecting `wait_minutes` pauses.
- `--phase [int]`: Starting phase (1-3).
- `--single-phase`: Execute only the specified starting phase and halt.
- `--project [name]`: Execute bumps or pushes exclusively for a targeted project name.

## Example: The Sequential Execution Pipeline

To run a safe end-to-end update sequence that stops if errors are detected:

```bash
repository-manager --validate --bump patch --maintain --push
```

1. **Validation**: Verifies static analysis and imports.
2. **Bump**: Bumps semantic versions.
3. **Maintain**: Propagates version changes in dependency trees.
4. **Push**: Pushes all projects upstream in parallel chunks, grouped by phase.

## Example: Targeting specific phases or projects

Run only Phase 2 pushing:
```bash
repository-manager --push --phase 2 --single-phase
```

Target only `agent-utilities` during a phased bump:
```bash
repository-manager --maintain --bump patch --project agent-utilities
```
