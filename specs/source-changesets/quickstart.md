# RM-SCM — Quickstart

Nothing is built yet. Once the first task lands, run the served proof of the import from the repository root:

```text
uv run pytest tests/source_changesets/test_import_served.py -q
```

The test starts the MCP server, opens a fastmcp `Client`, calls `rm_source_import` against a fixture repository and a recorded engine, and reads the receipt. The mirror and tool proofs run the same way: `uv run pytest tests/source_changesets/test_mirror_sync_served.py tests/source_changesets/test_dev_tools_served.py -q`.
