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

## Change-aware Start Phase (default)

Phased bump and push are **change-aware by default**: instead of blindly starting at
Phase 1, they first detect the **lowest phase that contains a repo with pending work**
and start there, skipping unchanged upstream phases and their inter-phase waits. This
applies everywhere — the CLI (`--maintain` / `--push`), the MCP tools
(`rm_workspace action="maintain"`, `rm_git action="phased_push"`), and the
`workspace-validator` skill's `auto_bump` / `auto_push`.

This is safe because phases are topologically ordered — a change in phase *N* can only
cascade downstream to phases `>= N` (dependency-pin propagation flows from upstream to
downstream, never the reverse). So:

- Edit something in **Phase 2** → the bump/push runs **Phase 2, 3, 4, …** and skips Phase 1.
- No repo has pending work → the bump/push is a **no-op** (nothing to do).

A repo is considered to have pending work when it is *not* both clean and in sync with
origin: i.e. it has uncommitted changes, an unpushed feature commit, or an unpushed
version bump awaiting delivery. An explicit starting phase (`--phase`) is honored as a
floor — the effective start is `max(explicit_phase, detected_phase)`.

**Opting out.** Pass `--no-auto-start` on the CLI, or `auto_start=False` to the MCP
tools / library functions, to force a start at the explicit `--phase` (default 1).
Change-aware start also **stands down automatically** for explicit-targeting requests —
when a `--project` / `project_filter` is given (bump and push), or `--force` (bump) —
since those deliberately bypass change detection.

## CLI Usage

The following flags control phased update and push sequences:

- `--validate`: Executes a full pre-release validation. If this fails, the next steps are aborted.
- `--bump [patch/minor/major]`: Executes a version bump.
- `--maintain`: Executes phased dependency updates across the workspace. Modifies `pyproject.toml` automatically based on dependency tree.
- `--push`: Executes a parallelized Git Push sequence per-phase, respecting `wait_minutes` pauses.
- `--phase [int]`: Starting phase (1-3). Acts as a floor under change-aware start.
- `--no-auto-start`: Opt out of change-aware start; begin at `--phase` (default 1) instead of the lowest changed phase.
- `--single-phase`: Execute only the specified starting phase and halt.
- `--project [name]`: Execute bumps or pushes exclusively for a targeted project name (disables change-aware start).

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
