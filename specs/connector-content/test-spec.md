# CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT — Test specification

Governing spec: [spec.md](spec.md).

Proof level: served tests enter through the MCP server built by `create_mcp_server` and a fastmcp `Client`; the engine is a recorded engine or a fixture. No live engine is required for ordinary validation.

## Positive

- `tests/test_default_content_surfaces.py::test_default_content_surfaces`. The listings include the packaged skills, ontology and prompts. *(CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R001)*

## Planned

- `tests/test_index_repository_served.py::test_tool_returns_changed_files_only` (planned). Two edited files of a fixture repository produce a page of those files; refusals `unsupported_language` and `invalid_type_mode` leave no state. *(CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R002)*
- `tests/test_repository_hydrate_served.py::test_hydrate_indexes_a_fixture_repository_through_arbora` (planned). The fixture repository indexes through the arbora MCP path and the manifest reads back; `REPOSITORY_UNKNOWN` and `REF_UNRESOLVED` write nothing. *(CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R003)*
- `tests/test_graph_reads.py::test_tree_and_blame_read_from_the_engine` (planned). A recorded engine answers; a repository absent from the engine falls back to the clone. *(CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R004)*
