# Seam: source client calls from the connector tools

Provider: scion-sdk SCION-SCM-R001
Consumer: repository-manager RM-SCM-R003.1

## Wire shape

Each `rm_source_*` tool calls the matching method of the SDK `SourceClient` (`tree`, `blob`, `diff`, `history`, `blame`, `commit`, `land`, `revert`) with the engine client of the caller. The SDK serves nothing itself; the connector composes the client.

## Refusal codes

The closed reasons of `SCION-SCM-R001.4`: `unauthorized`, `not_found`, `conflict`, `path_denied`, `oversized`, `deadline_exceeded`, `transport_unavailable`. The connector passes each through unchanged.

## Authoritative types

The request and page types of the SDK package `scion_sdk.source`.

## Test vectors

`tests/source_changesets/test_dev_tools.py` (planned) holds the connector-side vectors; the SDK holds `tests/test_source_client.py`.
