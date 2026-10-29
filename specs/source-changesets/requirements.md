# RM-SCM — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A child row `.n` is one five-minute unit of work. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `RM-SCM-R001` | **Import git history into engine changesets (rollup).** repository-manager must import the full history of one repository into engine changesets. Each commit becomes one changeset with the same tree hash, per MEDULLA-SCM-R010. Children `.1` to `.6` deliver it. | Satisfied only when `.1` to `.6` are verified. |
| `RM-SCM-R001.1` | **Model.** Add `ImportPlan{repo, remote, refs, first_commit, last_commit}` and `ImportReceipt{repo, commits, changesets, tree_hash_matches}` in `repository_manager/source_changesets/models.py`. | A unit test round-trips both models. |
| `RM-SCM-R001.2` | **Behavior.** The import walks commits in topological order. It submits one changeset per commit through the SDK source client of SCION-SCM-R001. A re-run must import only new commits. Each changeset keeps the author, the message and the trailers. | A test imports a 50-commit fixture twice and asserts 50 changesets, then 0. |
| `RM-SCM-R001.3` | **Wiring.** The served tool `rm_source_import` calls the import. arbora serves it as a connector tool behind admission. | A served test calls `rm_source_import` and reads the receipt. |
| `RM-SCM-R001.4` | **Refusal.** A tree hash mismatch must stop the import with `IMPORT_TREE_MISMATCH` and keep prior changesets. A shallow clone must refuse with `IMPORT_SHALLOW_HISTORY`. | A test asserts each code and no partial changeset. |
| `RM-SCM-R001.5` | **Evidence.** The receipt names the repository, the commit range, the changeset count and the tree hash result for each commit. | A test reads the receipt fields. |
| `RM-SCM-R001.6` | **Hot-path proof.** The entry point is the served `rm_source_import`. The test imports a 1,000-commit fixture. It must finish under 120 s and 512 MiB. | `tests/source_changesets/test_import_served.py::test_import_full_history` passes. |
| `RM-SCM-R002` | **Mirror sync to the public git hosts (rollup).** repository-manager must export each landed changeset to the public and internal git mirrors. It must pull a commit made on a mirror back as a changeset. Children `.1` to `.6` deliver it. | Satisfied only when `.1` to `.6` are verified. |
| `RM-SCM-R002.1` | **Model.** Add `MirrorTarget{repo, host, remote, branch}` and `MirrorReceipt{changeset_id, commit_sha, tree_hash, host}` in `repository_manager/source_changesets/models.py`. | A unit test round-trips both models. |
| `RM-SCM-R002.2` | **Behavior.** After `source.land`, the sync pushes the exported commit to each mirror within 60 s. A commit made on a mirror imports as one changeset through R001. The commit tree hash must equal the changeset tree hash. | A test lands one changeset and pulls one mirror commit back. |
| `RM-SCM-R002.3` | **Wiring.** The sync reads landed changesets through the SDK source client. It uses the git bridge export of ARBORA-OPS-R851 for the push. | A served test lands through arbora and observes the mirror push. |
| `RM-SCM-R002.4` | **Refusal.** A mirror that holds a diverged commit must stop the sync with `MIRROR_DIVERGED`. The sync never force-pushes. | A test diverges a fixture mirror and asserts the code and no push. |
| `RM-SCM-R002.5` | **Evidence.** Each push writes one receipt per host with the changeset id, the commit SHA and the tree hash. | A test reads the receipts. |
| `RM-SCM-R002.6` | **Hot-path proof.** The entry point is a served `source.land`. The test observes the mirror commit within 60 s with an equal tree hash. | `tests/source_changesets/test_mirror_sync_served.py::test_land_then_mirror` passes. |
| `RM-SCM-R003` | **Served source development tools (rollup).** repository-manager must serve one tool per `source.*` operation: tree, blob, diff, history, blame, commit, land and revert. Each tool composes the SDK source client. Children `.1` to `.3` deliver it. | Satisfied only when `.1` to `.3` are verified. |
| `RM-SCM-R003.1` | **Tools.** Register `rm_source_tree`, `rm_source_blob`, `rm_source_diff`, `rm_source_history`, `rm_source_blame`, `rm_source_commit`, `rm_source_land` and `rm_source_revert`. Reads carry the read-only hint. Writes carry the destructive hint. | A test lists the eight tools and their hints. |
| `RM-SCM-R003.2` | **Refusal.** Each tool passes the typed refusal of SCION-SCM-R001.4 through unchanged. A `CONFLICT` refusal names the conflicting paths. | A test asserts a passed-through `CONFLICT` with paths. |
| `RM-SCM-R003.3` | **Hot-path proof.** The entry point is arbora MCP. An agent edits one file through `rm_source_commit` and lands it through `rm_source_land`, per MERISTEM-SCM-R001. | `tests/source_changesets/test_dev_tools_served.py::test_commit_and_land_through_arbora` passes. |
| `RM-SCM-R004` | **One migration wave step for one repository (rollup).** repository-manager must move one repository to engine-primary for a wave of PIPE-GATES-R006. It must also move the repository back to git-primary. Children `.1` to `.3` deliver it. | Satisfied only when `.1` to `.3` are verified. |
| `RM-SCM-R004.1` | **Behavior.** The step imports history (R001), starts the mirror sync (R002) and records the repository as engine-primary. The reverse step stops intake and records git-primary. | A test runs the step and the reverse step on a fixture repository. |
| `RM-SCM-R004.2` | **Refusal.** The step must refuse with `WAVE_ENTRY_UNMET` when the wave entry criteria of PIPE-GATES-R006 do not hold. | A test asserts the refusal for an unmet criterion. |
| `RM-SCM-R004.3` | **Evidence.** The step writes a receipt with the wave, the repository, the primary side and the import receipt digest. | `tests/source_changesets/test_wave_step.py::test_step_and_reverse_receipts` passes. |
