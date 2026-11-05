# Seam: git bridge fetch for mirror export

Provider: arbora ARBORA-OPS-R851
Consumer: repository-manager RM-SCM-R002.3

## Wire shape

The mirror sync fetches the exported commits over the read routes `GET /git/{repository}/info/refs?service=git-upload-pack` and `POST /git/{repository}/git-upload-pack` and pushes them to the mirrors itself; the bridge serves no `git-receive-pack`.

## Refusal codes

HTTP 403 for a principal lacking a read grant on any path, and HTTP 403 with reason `push_not_served` for a push to the bridge.

## Authoritative types

The pack stream of the bridge routes.

## Test vectors

`tests/source_changesets/test_mirror_sync_served.py` (planned) holds the connector-side vector; arbora holds `tests/api/test_git_bridge_served.py`.
