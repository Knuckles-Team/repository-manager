# Seam: repository hydration through the SDK transport

Provider: scion-sdk SCION-REPOSITORY-TRANSPORT-R001
Consumer: repository-manager CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R003

## Wire shape

The tool `rm_repository_hydrate(repository_id, ref)` builds a local Git provider for the registered repository, resolves the immutable revision and calls the SDK function `index_repository_snapshot` with the engine client of the caller. The result carries the revision, the batch count and the manifest digest. The connector never calls the engine index operation itself (`SCION-REPOSITORY-TRANSPORT-R002`).

## Refusal codes

`REPOSITORY_UNKNOWN`, `REF_UNRESOLVED`. The SDK transport raises `RepositoryTransportError` for a provider, authentication or revision mismatch.

## Authoritative types

`RepositoryIndexReceipt` and `RepositoryRevision` in the SDK package `scion_sdk.repository`.

## Test vectors

`tests/test_repository_hydrate_served.py` (planned) holds the connector-side vector; the SDK holds `tests/test_repository_transport.py`.
