# Seam: served source operations behind arbora

Provider: arbora ARBORA-OPS-R848
Consumer: repository-manager RM-SCM-R003.1

## Wire shape

arbora serves the `source.*` operations with admission per principal and per path ACL, and the repository-manager connector is the development-side consumer that calls them through arbora MCP (`MERISTEM-SCM-R001` edits through the tools).

## Refusal codes

Admission refusals are those of the served operations; the connector returns them unchanged.

## Authoritative types

The served operation schemas of `ARBORA-OPS-R848`.

## Test vectors

`tests/source_changesets/test_dev_tools_served.py` (planned) holds the connector-side vector; arbora holds `tests/api/test_source_ops.py`.
