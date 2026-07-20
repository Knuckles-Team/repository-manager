"""Native epistemic-graph ingestion for repository-management records.

All writes use the required ``agent_utilities.knowledge_graph.memory.native_ingest``
primitive. Nodes use canonical ``node_type`` and edges use canonical ``relationship``;
nodes and edges commit in one native transaction. Missing engine dependencies, rejected
records, conflicts, and transaction failures propagate as ``NativeIngestError``.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.knowledge_graph.memory.native_ingest import (
    ingest_documents as _native_ingest_documents,
)
from agent_utilities.knowledge_graph.memory.native_ingest import (
    ingest_entities as _native_ingest_entities,
)

logger = logging.getLogger("repository_manager.kg")

_SOURCE = "repository-manager"
_DOMAIN = "repository"


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write canonical typed nodes and relationships in one native transaction."""
    return _native_ingest_entities(
        entities, relationships, source=source, domain=domain, client=client, graph=graph
    )


def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Write text records as canonical Document nodes."""
    return _native_ingest_documents(
        documents, source=source, domain=domain, client=client, graph=graph
    )


def _repo_ext_id(ref: dict[str, Any]) -> str | None:
    """Stable external id for a repository ref: numeric id, else full_path/name."""
    rid = ref.get("id")
    if rid is not None:
        return str(rid)
    return ref.get("full_path") or ref.get("name") or None


def ingest_repositories(
    refs: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map VCS enumeration refs (``vcs_enumerator``) → ``:GitRepository`` nodes and ingest.

    Accepts the RepoRef dicts produced by ``enumerate_gitlab``/``enumerate_github``
    (keys: ``vcs``, ``full_path``, ``clone_url``, ``web_url``, ``default_branch``,
    ``last_activity_at``, ``archived``, ``head_sha``, ``id``).
    """
    entities: list[dict[str, Any]] = []
    for ref in refs or []:
        ext = _repo_ext_id(ref)
        if ext is None:
            continue
        entities.append(
            {
                "id": f"repository:GitRepository:{ext}",
                "node_type": "GitRepository",
                "name": ref.get("full_path") or ref.get("name"),
                "vcs": ref.get("vcs"),
                "fullPath": ref.get("full_path"),
                "cloneUrl": ref.get("clone_url"),
                "repoWebUrl": ref.get("web_url"),
                "defaultBranch": ref.get("default_branch") or None,
                "lastActivityAt": ref.get("last_activity_at") or None,
                "archived": ref.get("archived"),
                "headSha": ref.get("head_sha") or None,
                "externalToolId": ext,
            }
        )
    return ingest_entities(entities, None, client=client, graph=graph)


def ingest_worktrees(
    worktrees: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map worktree records (``WorktreeManager.list_worktrees`` / ``audit``) → ``:Worktree`` nodes.

    Accepts worktree dicts with keys: ``repo``, ``path``, ``branch``, ``head``,
    ``linked`` and (from audit) ``class``/``status``, ``dirty``, ``ahead``, ``behind``.
    Links each worktree to its ``:GitRepository`` via ``:worktreeOf``.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for wt in worktrees or []:
        path = wt.get("path")
        repo = wt.get("repo")
        if not path:
            continue
        wt_id = f"repository:Worktree:{path}"
        entities.append(
            {
                "id": wt_id,
                "node_type": "Worktree",
                "name": f"{repo}:{wt.get('branch')}" if repo else path,
                "worktreePath": path,
                "branchName": wt.get("branch"),
                "headSha": wt.get("head") or None,
                "worktreeStatus": wt.get("class") or wt.get("status"),
                "dirty": wt.get("dirty"),
                "aheadCount": wt.get("ahead"),
                "behindCount": wt.get("behind"),
                "externalToolId": path,
            }
        )
        if repo:
            repo_id = f"repository:GitRepository:{repo}"
            entities.append(
                {
                    "id": repo_id,
                    "node_type": "GitRepository",
                    "name": repo,
                    "externalToolId": repo,
                }
            )
            relationships.append(
                {"source": wt_id, "target": repo_id, "relationship": "worktreeOf"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_projects(
    projects: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int]:
    """Map managed workspace projects → ``:Project`` nodes and ingest.

    Accepts project dicts with keys: ``name``, ``path`` and optionally
    ``validation_status``/``class``, ``dirty``, ``ahead``, ``behind``. Each is linked
    to its ``:GitRepository`` (same name) via ``:projectOfRepository``.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for proj in projects or []:
        name = proj.get("name") or proj.get("repo")
        path = proj.get("path")
        ext = name or path
        if not ext:
            continue
        proj_id = f"repository:Project:{ext}"
        entities.append(
            {
                "id": proj_id,
                "node_type": "Project",
                "name": name,
                "projectPath": path,
                "validationStatus": proj.get("validation_status") or proj.get("class"),
                "dirty": proj.get("dirty"),
                "aheadCount": proj.get("ahead") or proj.get("ahead_origin"),
                "behindCount": proj.get("behind") or proj.get("behind_origin"),
                "externalToolId": str(ext),
            }
        )
        if name:
            repo_id = f"repository:GitRepository:{name}"
            entities.append(
                {
                    "id": repo_id,
                    "node_type": "GitRepository",
                    "name": name,
                    "externalToolId": name,
                }
            )
            relationships.append(
                {"source": proj_id, "target": repo_id, "relationship": "projectOfRepository"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)
