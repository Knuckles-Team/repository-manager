# Seam: connector identity and default content

Provider: scion-sdk SCION-CONNECTOR-CONTROL-R071
Consumer: repository-manager CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R001

## Wire shape

The connector declares its fleet name once as the single entry of the `scion_sdk.connector` entry-point group in `pyproject.toml`; the value is the top-level package `repository_manager`. The server passes `ConnectorContent(connector, package_root, package_version)` to `create_mcp_server`; the SDK registers `skill://` and `ontology://<connector>/` resources and one prompt per `prompts/*.json` of the package root (`SCION-CONNECTOR-CONTROL-R073`).

## Refusal codes

The SDK raises `ConnectorIdentityError` for zero or two declared names. The connector adds no refusal code.

## Authoritative types

`ConnectorContent` in the SDK package `scion_sdk.mcp.content`; the entry-point declaration in `pyproject.toml`.

## Test vectors

`tests/test_default_content_surfaces.py` holds the served vector.
