# Seam: git import and export fidelity

Provider: medulla MEDULLA-SCM-R010
Consumer: repository-manager RM-SCM-R001.2

## Wire shape

A changeset maps one-to-one to a git commit with an equal tree id, in the object formats `sha1` and `sha256`. The import preserves the author, the message and the trailers; the export rebuilds the identical commit.

## Refusal codes

`GIT_GRAPH_MALFORMED`, `GIT_MODE_UNSUPPORTED`, `GIT_EXPORT_TREE_READ_FAILED`.

## Authoritative types

The `ImportGitHistory` and `ExportGitHistory` methods and the `Changeset` git origin of the engine row.

## Test vectors

`tests/source_changesets/test_import.py` (planned) holds the connector-side vectors.
