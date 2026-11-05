# CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT — Quickstart

Run the served proof of the built row from the repository root:

```text
uv run pytest tests/test_default_content_surfaces.py -q
```

The test builds the server through `create_mcp_server`, opens a fastmcp `Client` and lists resources and prompts. The planned tests of the index and hydration rows run the same way once added: `uv run pytest tests/test_index_repository_served.py tests/test_repository_hydrate_served.py -q`.
