"""Data sources of this connector, declared by tool name.

Served live as ``connector://resources``.
"""

from __future__ import annotations

from typing import Any

from scion_sdk.ingest.live_declaration import source_meta

CONNECTOR = "repository-manager"

SOURCES: tuple[Any, ...] = (
    source_meta(
        tool="rm_worktree_list",
        source_id="repository-worktrees",
        record_type="Worktree",
        id_field="path",
        preset={"doc_type": "repository", "params_style": "args"},
        records_path="worktrees",
        server="repository-manager",
        text_field="repo",
        title_field="branch",
    ),
)
