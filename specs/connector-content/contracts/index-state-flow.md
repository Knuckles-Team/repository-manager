# Seam: incremental repository index state

Provider: repository-manager CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R002
Consumer: scion-sdk SCION-SOURCE-INGEST-R004.15

## Wire shape

The SDK worker passes the previous index state to the tool `index_repository` through `McpToolSourceAdapter.extract`; the tool returns index pages for changed files only, in `syntax` or `typed` mode, plus the new state. The connector owns Git access; the SDK owns state, paging and receipts. The state type name is owned by the SDK row.

## Refusal codes

`unsupported_language`, `invalid_type_mode`.

## Authoritative types

The SDK source-adapter port and index page types named by `SCION-SOURCE-INGEST-R004.15`.

## Test vectors

`tests/test_index_repository_served.py` (planned) holds the connector-side vector; the SDK holds `tests/test_repository_effect_contract.py::test_state_flows_worker_to_adapter`.
