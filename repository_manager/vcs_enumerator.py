#!/usr/bin/python
from __future__ import annotations

"""Remote VCS enumeration for enterprise-scale ingestion (CONCEPT:AU-KG.ontology.populated-at-import-real-3).

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_utilities.core.transport_security import (
    ResolvedTLSProfile,
    resolve_configured_tls_profile,
)

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


def _ensure_client(
    client: Any, tls_profile: ResolvedTLSProfile | None, service: str
) -> tuple[Any, bool, bool, ResolvedTLSProfile | None]:
    """Build (or pass through) the injected httpx-style client + TLS profile.

    Returns ``(client, owns_client, owns_profile, tls_profile)``. ``client`` is
    ``None`` only when a live client was needed but ``httpx`` isn't installed —
    callers must check for that and return early without entering a try/finally.
    """
    owns_client = client is None
    owns_profile = tls_profile is None
    if owns_client:
        if not _HTTPX:
            return None, owns_client, owns_profile, tls_profile
        tls_profile = tls_profile or resolve_configured_tls_profile(service)
        client = httpx.Client(timeout=30.0, **tls_profile.httpx_kwargs())
    return client, owns_client, owns_profile, tls_profile


def _cleanup_client(
    client: Any,
    owns_client: bool,
    owns_profile: bool,
    tls_profile: ResolvedTLSProfile | None,
) -> None:
    if owns_client:
        client.close()
    if owns_profile and tls_profile is not None:
        tls_profile.cleanup()


@dataclass
class _GitlabEnumOptions:
    groups: Sequence[str | int] | None
    include_subgroups: bool
    archived: bool
    updated_after: str | None
    max_repos: int | None


def _gitlab_targets(base: str, groups: Sequence[str | int] | None) -> list[str]:
    if groups:
        return [f"{base.rstrip('/')}/api/v4/groups/{g}/projects" for g in groups]
    return [f"{base.rstrip('/')}/api/v4/projects"]


def _gitlab_page_params(id_after: int, opts: _GitlabEnumOptions) -> dict[str, Any]:
    params: dict[str, Any] = {
        "pagination": "keyset",
        "per_page": 100,
        "order_by": "id",
        "sort": "asc",
        "id_after": id_after,
        "simple": "true",
        "archived": str(opts.archived).lower(),
    }
    if opts.groups:
        params["include_subgroups"] = str(opts.include_subgroups).lower()
    else:
        params["membership"] = "true"
    if opts.updated_after:
        params["last_activity_after"] = opts.updated_after
    return params


def _gitlab_accumulate_batch(
    batch: list[dict[str, Any]], opts: _GitlabEnumOptions, out: list[dict[str, Any]]
) -> bool:
    """Normalize+filter one page of GitLab projects into ``out``.

    Returns True once ``max_repos`` is hit.
    """
    for p in batch:
        ref = _gitlab_ref(p)
        if ref["archived"] and not opts.archived:
            continue
        out.append(ref)
        if opts.max_repos is not None and len(out) >= opts.max_repos:
            return True
    return False


def _gitlab_enumerate_one(
    client: Any,
    headers: dict[str, str],
    url: str,
    opts: _GitlabEnumOptions,
    out: list[dict[str, Any]],
) -> bool:
    """Paginate one GitLab target URL, appending refs to ``out``.

    Returns True once ``max_repos`` is hit (signal to stop enumerating further
    targets entirely, matching the original single-function early return).
    """
    id_after = 0
    while True:
        params = _gitlab_page_params(id_after, opts)
        resp = client.get(url, headers=headers, params=params)
        if getattr(resp, "status_code", 0) != 200:
            return False
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            return False
        if _gitlab_accumulate_batch(batch, opts, out):
            return True
        id_after = max(int(p.get("id", 0)) for p in batch)
        if len(batch) < 100:
            return False


def _gitlab_enumerate_targets(
    client: Any, headers: dict[str, str], targets: list[str], opts: _GitlabEnumOptions
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for url in targets:
        if _gitlab_enumerate_one(client, headers, url, opts, out):
            break
    return out


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
    tls_profile: ResolvedTLSProfile | None = None,
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
    client, owns_client, owns_profile, tls_profile = _ensure_client(
        client, tls_profile, "gitlab"
    )
    if client is None:
        return []
    headers = {"PRIVATE-TOKEN": tok} if tok else {}
    targets = _gitlab_targets(base, groups)
    opts = _GitlabEnumOptions(
        groups, include_subgroups, archived, updated_after, max_repos
    )
    try:
        out = _gitlab_enumerate_targets(client, headers, targets, opts)
    finally:
        _cleanup_client(client, owns_client, owns_profile, tls_profile)
    return out


@dataclass
class _GithubEnumOptions:
    user: bool
    archived: bool
    max_repos: int | None


def _github_headers(tok: str | None) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _github_targets(orgs: list[str] | None) -> list[str]:
    if orgs:
        return [f"https://api.github.com/orgs/{o}/repos" for o in orgs]
    return ["https://api.github.com/user/repos"]


def _github_page_params(
    page: int, url: str, opts: _GithubEnumOptions
) -> dict[str, Any]:
    params: dict[str, Any] = {"per_page": 100, "page": page}
    if opts.user and "user/repos" in url:
        params["affiliation"] = "owner,collaborator,organization_member"
    return params


def _github_accumulate_batch(
    batch: list[dict[str, Any]], opts: _GithubEnumOptions, out: list[dict[str, Any]]
) -> bool:
    """Normalize+filter one page of GitHub repos into ``out``.

    Returns True once ``max_repos`` is hit.
    """
    for r in batch:
        ref = _github_ref(r)
        if ref["archived"] and not opts.archived:
            continue
        out.append(ref)
        if opts.max_repos is not None and len(out) >= opts.max_repos:
            return True
    return False


def _github_enumerate_one(
    client: Any,
    headers: dict[str, str],
    url: str,
    opts: _GithubEnumOptions,
    out: list[dict[str, Any]],
) -> bool:
    """Paginate one GitHub target URL, appending refs to ``out``.

    Returns True once ``max_repos`` is hit (signal to stop enumerating further
    targets entirely, matching the original single-function early return).
    """
    page = 1
    while True:
        params = _github_page_params(page, url, opts)
        resp = client.get(url, headers=headers, params=params)
        if getattr(resp, "status_code", 0) != 200:
            return False
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            return False
        if _github_accumulate_batch(batch, opts, out):
            return True
        if len(batch) < 100:
            return False
        page += 1


def _github_enumerate_targets(
    client: Any, headers: dict[str, str], targets: list[str], opts: _GithubEnumOptions
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for url in targets:
        if _github_enumerate_one(client, headers, url, opts, out):
            break
    return out


def enumerate_github(
    token: str | None = None,
    *,
    orgs: list[str] | None = None,
    user: bool = False,
    archived: bool = False,
    max_repos: int | None = None,
    client: Any = None,
    tls_profile: ResolvedTLSProfile | None = None,
) -> list[dict[str, Any]]:
    """Enumerate GitHub repos per org (or the authenticated user) via pagination."""
    tok = _github_token(token)
    client, owns_client, owns_profile, tls_profile = _ensure_client(
        client, tls_profile, "github"
    )
    if client is None:
        return []
    headers = _github_headers(tok)
    targets = _github_targets(orgs)
    opts = _GithubEnumOptions(user, archived, max_repos)
    try:
        out = _github_enumerate_targets(client, headers, targets, opts)
    finally:
        _cleanup_client(client, owns_client, owns_profile, tls_profile)
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
