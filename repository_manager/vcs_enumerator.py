#!/usr/bin/python
from __future__ import annotations

"""Remote VCS enumeration for enterprise-scale ingestion (CONCEPT:KG-2.49).

Lists every repository across a GitLab instance or GitHub org so the KG-side
batch ingestor (``agent_utilities.knowledge_graph.ingestion.batch_orchestrator``)
can fan deep-code ingestion into the durable queue. This is the enumeration gap:
the engine and queue already scale, but nothing listed 20k+ remote repos.

Scale specifics:
  - **GitLab uses keyset pagination** (``pagination=keyset`` + ``id_after``
    cursor), mandatory past ~10k offset where page-number pagination breaks.
    Enumerates per group (``include_subgroups``) or the whole instance.
  - **GitHub uses page pagination** with ``per_page=100`` per org, stopping when a
    short page is returned.

The HTTP client is injectable (any object with a ``.get(url, headers=, params=)``
returning an httpx-style response) so this is unit-testable offline; the default
constructs an ``httpx.Client``. Output is a list of normalized ``dict`` refs
(matching ``batch_orchestrator.RepoRef`` fields) and/or a JSON manifest written
under ``~/workspace/reports/``.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:  # optional at import — only needed for the live (non-injected) path
    import httpx

    _HTTPX = True
except ImportError:  # pragma: no cover - exercised only without httpx
    _HTTPX = False


def _gitlab_creds(
    base_url: str | None, token: str | None
) -> tuple[str | None, str | None]:
    return (
        base_url or os.getenv("GITLAB_URL") or os.getenv("GITLAB_HOST"),
        token or os.getenv("GITLAB_TOKEN") or os.getenv("GITLAB_PRIVATE_TOKEN"),
    )


def _github_token(token: str | None) -> str | None:
    return token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _gitlab_ref(p: dict[str, Any]) -> dict[str, Any]:
    """Normalize a GitLab project JSON into a RepoRef dict."""
    return {
        "vcs": "gitlab",
        "full_path": p.get("path_with_namespace", ""),
        "clone_url": p.get("http_url_to_repo", ""),
        "web_url": p.get("web_url", ""),
        "default_branch": p.get("default_branch", "") or "",
        "last_activity_at": p.get("last_activity_at", "") or "",
        "archived": bool(p.get("archived", False)),
        # HEAD sha is filled by the clone step; the listing doesn't expose it.
        "head_sha": "",
        "id": p.get("id"),
    }


def _github_ref(r: dict[str, Any]) -> dict[str, Any]:
    """Normalize a GitHub repo JSON into a RepoRef dict."""
    return {
        "vcs": "github",
        "full_path": r.get("full_name", ""),
        "clone_url": r.get("clone_url", ""),
        "web_url": r.get("html_url", ""),
        "default_branch": r.get("default_branch", "") or "",
        "last_activity_at": r.get("pushed_at", "") or "",
        "archived": bool(r.get("archived", False)),
        "head_sha": "",
    }


def enumerate_gitlab(
    base_url: str | None = None,
    token: str | None = None,
    *,
    groups: Sequence[str | int] | None = None,
    include_subgroups: bool = True,
    archived: bool = False,
    updated_after: str | None = None,
    max_repos: int | None = None,
    client: Any = None,
    verify_ssl: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate GitLab projects via keyset pagination.

    Args:
        base_url / token: GitLab instance + PAT (env fallbacks).
        groups: group ids/paths to enumerate (with subgroups); ``None`` =
            the whole instance (``/projects?membership``).
        include_subgroups: descend into subgroups when listing per group.
        archived: include archived projects (default excluded).
        updated_after: ISO timestamp for incremental enumeration (server-side).
        max_repos: cap total refs (``None`` = all).
        client: injected httpx-style client (default builds one).
    """
    base, tok = _gitlab_creds(base_url, token)
    if not base:
        return []
    owns_client = client is None
    if owns_client:
        if not _HTTPX:
            return []
        client = httpx.Client(verify=verify_ssl, timeout=30.0)
    headers = {"PRIVATE-TOKEN": tok} if tok else {}
    targets: list[str] = (
        [f"{base.rstrip('/')}/api/v4/groups/{g}/projects" for g in groups]
        if groups
        else [f"{base.rstrip('/')}/api/v4/projects"]
    )
    out: list[dict[str, Any]] = []
    try:
        for url in targets:
            id_after = 0
            while True:
                params: dict[str, Any] = {
                    "pagination": "keyset",
                    "per_page": 100,
                    "order_by": "id",
                    "sort": "asc",
                    "id_after": id_after,
                    "simple": "true",
                    "archived": str(archived).lower(),
                }
                if groups:
                    params["include_subgroups"] = str(include_subgroups).lower()
                else:
                    params["membership"] = "true"
                if updated_after:
                    params["last_activity_after"] = updated_after
                resp = client.get(url, headers=headers, params=params)
                if getattr(resp, "status_code", 0) != 200:
                    break
                batch = resp.json()
                if not isinstance(batch, list) or not batch:
                    break
                for p in batch:
                    ref = _gitlab_ref(p)
                    if ref["archived"] and not archived:
                        continue
                    out.append(ref)
                    if max_repos is not None and len(out) >= max_repos:
                        return out
                id_after = max(int(p.get("id", 0)) for p in batch)
                if len(batch) < 100:
                    break
    finally:
        if owns_client:
            client.close()
    return out


def enumerate_github(
    token: str | None = None,
    *,
    orgs: list[str] | None = None,
    user: bool = False,
    archived: bool = False,
    max_repos: int | None = None,
    client: Any = None,
) -> list[dict[str, Any]]:
    """Enumerate GitHub repos per org (or the authenticated user) via pagination."""
    tok = _github_token(token)
    owns_client = client is None
    if owns_client:
        if not _HTTPX:
            return []
        client = httpx.Client(timeout=30.0)
    headers = {"Accept": "application/vnd.github+json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    targets = (
        [f"https://api.github.com/orgs/{o}/repos" for o in orgs]
        if orgs
        else ["https://api.github.com/user/repos"]
    )
    out: list[dict[str, Any]] = []
    try:
        for url in targets:
            page = 1
            while True:
                params: dict[str, Any] = {"per_page": 100, "page": page}
                if user and "user/repos" in url:
                    params["affiliation"] = "owner,collaborator,organization_member"
                resp = client.get(url, headers=headers, params=params)
                if getattr(resp, "status_code", 0) != 200:
                    break
                batch = resp.json()
                if not isinstance(batch, list) or not batch:
                    break
                for r in batch:
                    ref = _github_ref(r)
                    if ref["archived"] and not archived:
                        continue
                    out.append(ref)
                    if max_repos is not None and len(out) >= max_repos:
                        return out
                if len(batch) < 100:
                    break
                page += 1
    finally:
        if owns_client:
            client.close()
    return out


def write_manifest(
    refs: list[dict[str, Any]], run_id: str, out_dir: str | None = None
) -> str:
    """Write the enumerated refs as a JSON ingest manifest under reports/.

    Returns the manifest path. Never writes to a repo root (AGENTS hygiene).
    """
    base = Path(
        out_dir
        or os.getenv("WORKSPACE_REPORTS")
        or (Path.home() / "workspace" / "reports")
    )
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"vcs_enumeration_{run_id}.json"
    path.write_text(
        json.dumps({"run_id": run_id, "count": len(refs), "repos": refs}, indent=2)
    )
    return str(path)


__all__ = ["enumerate_gitlab", "enumerate_github", "write_manifest"]
