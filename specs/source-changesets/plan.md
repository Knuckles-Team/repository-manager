# RM-SCM — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

A planned package `repository_manager/source_changesets/` imports Git history into engine changesets, pushes each landed changeset to the git mirrors, and runs one wave step per repository; a planned registrar `repository_manager/mcp_tools/source.py` serves one tool per `source.*` operation. Every engine effect goes through the SDK `SourceClient`; every Git process goes through `Git.git_action`. Nothing exists yet; the rows carry the exact attach points. Reuse candidates found by searching the code first: `repository_manager/repository_manager.py` (`Git.git_action` at :2462, the audited subprocess runner), `repository_manager/mcp_tools/registry.py` (the `MCP_TOOL_REGISTRY` seam), `repository_manager/mcp_tools/merge_queue.py` (a registrar to copy), `repository_manager/materialization/models.py` (the frozen-model pattern), `repository_manager/development/serialization.py` (`canonical_digest`).

```text
arbora MCP -> repository-manager connector   repository_manager/mcp_server.py:973
  -> MCP_TOOL_REGISTRY entry register_source_tools   repository_manager/mcp_tools/registry.py:105 (planned entry)
  -> rm_source_import / rm_source_land / rm_source_wave_step   repository_manager/mcp_tools/source.py (planned)
  -> import_history / mirror_changeset / wave_step   repository_manager/source_changesets/ (planned)
  -> SourceClient (SDK) and Git.git_action
```

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: `scion-sdk[mcp]` (the source client and the tool registration), pydantic, the system `git` binary.
- **Storage**: none in this repository; changesets live in the engine and commits live on the mirrors.
- **Testing**: `pytest` under `tests/source_changesets/` with temporary Git repositories, a recorded engine and a local bare repository standing for a mirror.
- **Target Platform**: Linux and macOS hosts running the MCP server.
- **Project Type**: Connector.
- **Performance Goals**: a 1,000-commit import under 120 s and 512 MiB; a mirror push within 60 s of a landing.
- **Constraints**: the sync never force-pushes; a refusal leaves no partial changeset; the connector holds no engine call of its own.
- **Scale/Scope**: 4 rollups and 18 leaf rows, 4 user stories, 4 `FR-` labels and 2 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. The SDK client and `Git.git_action` are reused; no second Git runner or engine client.
- **II. Complete specification before construction**: pass. Each row names its caller chain, shape and test; none is built.
- **III. Observable integration and evidence**: pass. Hot-path rows enter through the MCP client.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
repository_manager/source_changesets/models.py     new: ImportPlan, ImportReceipt, MirrorTarget, MirrorReceipt, refusal enums (planned)
repository_manager/source_changesets/importer.py   new: import_history (planned)
repository_manager/source_changesets/mirror.py     new: mirror_changeset (planned)
repository_manager/source_changesets/wave.py       new: wave_step (planned)
repository_manager/mcp_tools/source.py             new: register_source_tools (planned)
repository_manager/mcp_tools/registry.py           existing: MCP_TOOL_REGISTRY gains one entry (planned)
repository_manager/connectors/tool_schema_fingerprints.json  existing: gains the new tools (planned)
tests/source_changesets/                           new: unit and served tests (planned)
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The open choices (import through per-commit `commit` calls or through the engine `ImportGitHistory` stream, how landings made outside `rm_source_land` reach the sync, where the primary side is stored, how wave criteria reach the connector, and the module names) are owner decisions recorded in the pull request.

## Phase 1: Design

1. **Import.** Walk commits in topological order, submit one changeset per commit, compare tree ids after each, keep the author, message and trailers. A re-run imports only commits the engine does not hold.
2. **Mirror sync.** After a landed receipt, push the exported commit to each mirror without force; a commit that arrives on a mirror imports back through the same path.
3. **Development tools.** One thin tool per `source.*` operation; reads carry the read-only hint, writes the destructive hint; refusals pass through unchanged.
4. **Wave step.** Import, then enable the mirror sync, then record the primary side; the reverse stops intake.
5. **Typed refusals.** `IMPORT_TREE_MISMATCH`, `IMPORT_SHALLOW_HISTORY`, `MIRROR_DIVERGED` and `WAVE_ENTRY_UNMET`.

Data entities are the planned frozen models in `models.py`; the SDK types are not redefined.

## Phase 2: Task approach

`tasks.md` holds one task per leaf row, producer first: models, import, mirror, tools, wave. The evidence tasks run in parallel with the refusal tasks of the same rollup.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests/source_changesets` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit` and `vulture` pass without a suppression.
- The tool-schema fingerprint file and the connector standard hook accept the new tools.
- The no-duplicate-authority check finds no second Git runner and no second engine call.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

The chain starts at arbora MCP calling the connector built at `repository_manager/mcp_server.py:973`; `register_split_action_tools(mcp, list(MCP_TOOL_REGISTRY))` (:994) registers the planned `register_source_tools` entry (`repository_manager/mcp_tools/registry.py:105`), whose tools call `import_history`, `mirror_changeset` and `wave_step`, which end at the SDK `SourceClient` (`scion_sdk/source/client.py:84`) and `Git.git_action` (`repository_manager/repository_manager.py:2462`). Effects: engine changesets through the SDK, commits on the mirrors through `git push`. Obsolete paths to delete: none.

## Cross-repo seams

- `SCION-SCM-R001` source client calls, provider: scion-sdk, consumer: `RM-SCM-R003.1`.
- `ARBORA-OPS-R848` served source operations, provider: arbora, consumer: `RM-SCM-R003.1`.
- `ARBORA-OPS-R851` git bridge fetch, provider: arbora, consumer: `RM-SCM-R002.3`.
- `MEDULLA-SCM-R010` git import and export fidelity, provider: medulla, consumer: `RM-SCM-R001.2`.
- `PIPE-GATES-R006` migration wave entry criteria, provider: pipelines, consumer: `RM-SCM-R004.2`.
