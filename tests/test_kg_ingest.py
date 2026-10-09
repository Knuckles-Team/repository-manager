"""Native epistemic-graph typed-node ingestion — SDK ingest facade coverage.

Exercises the real ``ingest_entities`` / ``ingest_repositories`` / ``ingest_worktrees`` /
``ingest_projects`` seams against a fake ``agent_connector_sdk.ingest`` transport (no engine
required), asserting the generated request's records/relationships and the record →
:GitRepository / :Worktree / :Project mappings.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_connector_sdk.ingest import IngestError, KnowledgeIngest

from repository_manager.kg_ingest import (
    ingest_entities,
    ingest_projects,
    ingest_repositories,
    ingest_worktrees,
)


class _FakeTransport:
    def __init__(self):
        self.requests = []

    async def source_status(self, connector, stream):
        return SimpleNamespace(accepted_checkpoint=None)

    async def submit(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            affected_count=len(request.records),
            relationship_count=len(request.relationships),
        )

    async def store_blob(self, data):
        raise AssertionError("this connector's ingestion carries no media")


@pytest.fixture
def ingest():
    transport = _FakeTransport()
    return KnowledgeIngest(transport, loop=None), transport


async def test_ingest_entities_writes_nodes_and_edges(ingest):
    service, transport = ingest
    res = await ingest_entities(
        [
            {"id": "a", "node_type": "GitRepository", "name": "p"},
            {"id": "b", "node_type": "Worktree"},
        ],
        [{"source": "b", "target": "a", "relationship": "worktreeOf"}],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    assert {r.record_id for r in request.records} == {"a", "b"}
    assert request.relationships[0].source.record_id == "b"
    assert request.relationships[0].target.record_id == "a"


async def test_ingest_repositories_maps_gitrepository(ingest):
    service, transport = ingest
    res = await ingest_repositories(
        [
            {
                "vcs": "gitlab",
                "full_path": "grp/demo",
                "clone_url": "https://gl/grp/demo.git",
                "web_url": "https://gl/grp/demo",
                "default_branch": "main",
                "last_activity_at": "2026-07-01T00:00:00Z",
                "archived": False,
                "head_sha": "",
                "id": 42,
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 1, "edges": 0}
    record = transport.requests[0].records[0]
    assert record.record_id == "repository:GitRepository:42"
    assert record.payload["vcs"] == "gitlab"
    assert record.payload["fullPath"] == "grp/demo"
    assert record.payload["cloneUrl"] == "https://gl/grp/demo.git"
    assert record.payload["defaultBranch"] == "main"
    assert record.payload["externalToolId"] == "42"
    # empty head_sha is dropped (falsy-filtered in the mapping layer)
    assert not record.payload.get("headSha")


async def test_ingest_repositories_falls_back_to_full_path_id(ingest):
    service, transport = ingest
    await ingest_repositories(
        [{"vcs": "github", "full_path": "owner/repo", "clone_url": "x"}],
        ingest=service,
    )
    assert transport.requests[0].records[0].record_id == "repository:GitRepository:owner/repo"


async def test_ingest_worktrees_maps_and_links_repo(ingest):
    service, transport = ingest
    res = await ingest_worktrees(
        [
            {
                "repo": "agent-utilities",
                "path": "worktree://agent-utilities/feat-x",
                "branch": "feat/x",
                "head": "abc1234567",
                "class": "active",
                "dirty": True,
                "ahead": 3,
                "behind": 0,
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    wt = next(
        r
        for r in request.records
        if r.record_id == "repository:Worktree:worktree://agent-utilities/feat-x"
    )
    assert wt.payload["branchName"] == "feat/x"
    assert wt.payload["worktreeStatus"] == "active"
    assert wt.payload["dirty"] is True
    assert wt.payload["aheadCount"] == 3
    assert any(
        r.record_id == "repository:GitRepository:agent-utilities" for r in request.records
    )
    assert request.relationships[0].source.record_id == (
        "repository:Worktree:worktree://agent-utilities/feat-x"
    )
    assert request.relationships[0].target.record_id == (
        "repository:GitRepository:agent-utilities"
    )


async def test_ingest_projects_maps_and_links_repo(ingest):
    service, transport = ingest
    res = await ingest_projects(
        [
            {
                "name": "gitlab-api",
                "path": "repository://gitlab-api",
                "class": "clean",
                "dirty": False,
                "ahead_origin": 0,
            }
        ],
        ingest=service,
    )
    assert res == {"nodes": 2, "edges": 1}
    request = transport.requests[0]
    proj = next(r for r in request.records if r.record_id == "repository:Project:gitlab-api")
    assert proj.payload["validationStatus"] == "clean"
    assert proj.payload["projectPath"].endswith("gitlab-api")
    assert request.relationships[0].source.record_id == "repository:Project:gitlab-api"
    assert request.relationships[0].target.record_id == "repository:GitRepository:gitlab-api"


async def test_empty_ingest_entities_is_rejected(ingest):
    service, _transport = ingest
    with pytest.raises(IngestError, match="at least one entity"):
        await ingest_entities([], ingest=service)
