"""Native epistemic-graph ingestion for repository-manager records (typed graph nodes).

CONCEPT:AU-KG.ingest.enterprise-source-extractor. The repository-manager natively pushes
its data into the ONE epistemic-graph knowledge graph as **typed OWL nodes**
(``:GitRepository``, ``:Worktree``, ``:Project``) + links, using the lightweight engine
client (``GraphComputeEngine()._client`` + ``txn``) — the same fast client the blob
``MediaStore`` uses, NOT the heavy in-process ingestion engine.

Everything is dependency-/engine-guarded: with no agent-utilities KG stack or no reachable
engine, every entry point **no-ops** (returns ``None``), so the connector keeps working with
zero KG infrastructure. Nodes carry the shared provenance (``domain``/``source``) and match
the classes federated by ``repository_manager.ontology`` (``repository.ttl``). Node ids follow
``repository:<class>:<externalId>``.

This is a thin mapper: it prefers the shared primitive
``agent_utilities.knowledge_graph.memory.native_ingest`` when present, and falls back to a
self-contained txn write when that primitive is not in the installed agent_utilities.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("repository_manager.kg")

_SOURCE = "repository-manager"
_DOMAIN = "repository"
_DEFAULT_GRAPH = "__commons__"


def _client() -> tuple[Any | None, str]:
    """Return ``(engine_client, graph_name)`` or ``(None, "")`` when unavailable."""
    try:
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )
    except Exception as e:  # noqa: BLE001 — KG stack absent
        logger.debug("KG ingest unavailable (import): %s", e)
        return None, ""
    try:
        engine = GraphComputeEngine()
        client = getattr(engine, "_client", None)
        if client is None:
            return None, ""
        graph = getattr(engine, "graph_name", None) or _DEFAULT_GRAPH
        return client, graph
    except Exception as e:  # noqa: BLE001 — engine unreachable
        logger.debug("KG ingest: engine unreachable: %s", e)
        return None, ""


def _write_nodes(
    client: Any,
    graph: str,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None,
) -> dict[str, int] | None:
    """Stamp provenance, MERGE nodes in one txn, then add edges. Best-effort."""
    nodes = [n for n in nodes if n.get("id")]
    if not nodes:
        return None
    try:
        txn = client.txn.begin(graph=graph)
        for node in nodes:
            props = {k: v for k, v in node.items() if k != "id" and v is not None}
            props.setdefault("source", _SOURCE)
            props.setdefault("domain", _DOMAIN)
            client.txn.add_node(txn, node["id"], props)
        committed = client.txn.commit(txn)
    except Exception as e:  # noqa: BLE001 — engine/txn failure is non-fatal
        logger.warning("KG ingest: txn failed: %s", e)
        return None
    if not committed:
        logger.warning("KG ingest: txn not committed (conflict)")
        return None

    edges = 0
    for rel in relationships or []:
        try:
            client.edges.add(
                rel["source"], rel["target"], {"type": rel.get("type", "RELATED")}
            )
            edges += 1
        except Exception as e:  # noqa: BLE001 — pure edge link, best-effort
            logger.debug("KG ingest: edge skipped: %s", e)

    logger.info("KG ingest: wrote %d nodes, %d edges", len(nodes), edges)
    return {"nodes": len(nodes), "edges": edges}


def ingest_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write typed OWL nodes (+ edges) into epistemic-graph via the fast engine client.

    ``entities``: ``[{"id":..., "type":<owl:Class>, ...props}]``.
    ``relationships``: ``[{"source":id, "target":id, "type":<link>}]``.
    Returns ``{"nodes":n, "edges":m}`` or ``None`` (no engine / failure; never raises).
    Prefers the shared ``native_ingest`` primitive; otherwise falls back to the local
    txn path. ``client``/``graph`` may be injected (tests).
    """
    entities = [e for e in (entities or []) if e.get("id")]
    if not entities:
        return None
    # Prefer the shared primitive when present and no client was injected.
    if client is None:
        try:
            from agent_utilities.knowledge_graph.memory.native_ingest import (
                ingest_entities as _shared,
            )

            return _shared(
                entities, relationships, source=source, domain=domain, graph=graph
            )
        except Exception as e:  # noqa: BLE001 — primitive absent; use local fallback
            logger.debug("KG ingest: shared primitive unavailable: %s", e)
        client, graph = _client()
    if client is None:
        return None
    return _write_nodes(client, graph or _DEFAULT_GRAPH, entities, relationships)


def ingest_documents(
    documents: list[dict[str, Any]],
    *,
    source: str = _SOURCE,
    domain: str = _DOMAIN,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
    """Write text records (e.g. READMEs) as ``:Document`` nodes for semantic search.

    Each doc: ``{"id":..., "text":..., "title"?:..., "source_uri"?:..., ...props}``.
    Prefers the shared primitive; falls back to a local ``:Document`` write.
    """
    documents = [
        d
        for d in (documents or [])
        if d.get("id") and (d.get("text") or d.get("content"))
    ]
    if not documents:
        return None
    if client is None:
        try:
            from agent_utilities.knowledge_graph.memory.native_ingest import (
                ingest_documents as _shared,
            )

            return _shared(documents, source=source, domain=domain, graph=graph)
        except Exception as e:  # noqa: BLE001 — primitive absent; use local fallback
            logger.debug("KG ingest: shared primitive unavailable: %s", e)
        client, graph = _client()
    if client is None:
        return None
    nodes: list[dict[str, Any]] = []
    for doc in documents:
        text = doc.get("text") or doc.get("content")
        node = {k: v for k, v in doc.items() if k != "content" and v is not None}
        node["type"] = "Document"
        node["text"] = text
        nodes.append(node)
    return _write_nodes(client, graph or _DEFAULT_GRAPH, nodes, None)


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
) -> dict[str, int] | None:
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
                "type": "GitRepository",
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
) -> dict[str, int] | None:
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
                "type": "Worktree",
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
                    "type": "GitRepository",
                    "name": repo,
                    "externalToolId": repo,
                }
            )
            relationships.append(
                {"source": wt_id, "target": repo_id, "type": "worktreeOf"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)


def ingest_projects(
    projects: list[dict[str, Any]],
    *,
    client: Any | None = None,
    graph: str | None = None,
) -> dict[str, int] | None:
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
                "type": "Project",
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
                    "type": "GitRepository",
                    "name": name,
                    "externalToolId": name,
                }
            )
            relationships.append(
                {"source": proj_id, "target": repo_id, "type": "projectOfRepository"}
            )
    return ingest_entities(entities, relationships, client=client, graph=graph)
