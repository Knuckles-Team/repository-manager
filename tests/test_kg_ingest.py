"""Native epistemic-graph typed-node ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_repositories`` / ``ingest_worktrees`` /
``ingest_projects`` seams with a fake engine client (no engine required), asserting the txn
add_node/commit + edge calls and the record → :GitRepository / :Worktree / :Project mappings.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

import pytest
from agent_utilities.knowledge_graph.memory.native_ingest import NativeIngestError

from repository_manager.kg_ingest import (
    ingest_entities,
    ingest_projects,
    ingest_repositories,
    ingest_worktrees,
)


class _FakeTxn:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.committed = False

    def begin(self, graph=None):
        self.graph = graph
        return "txn-1"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = props

    def add_edge(self, txn, source, target, props):
        self.edges.append((source, target, props))

    def commit(self, txn):
        self.committed = True
        return True


class _FakeClient:
    def __init__(self):
        self.txn = _FakeTxn()


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
        [
            {"id": "a", "node_type": "GitRepository", "name": "p"},
            {"id": "b", "node_type": "Worktree"},
        ],
        [{"source": "b", "target": "a", "relationship": "worktreeOf"}],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    assert c.txn.committed is True
    assert set(c.txn.nodes) == {"a", "b"}
    # provenance is stamped
    assert c.txn.nodes["a"]["source"] == "repository-manager"
    assert c.txn.nodes["a"]["domain"] == "repository"
    assert c.txn.edges == [("b", "a", {"relationship": "worktreeOf"})]


def test_ingest_repositories_maps_gitrepository():
    c = _FakeClient()
    res = ingest_repositories(
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
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 1, "edges": 0}
    node = c.txn.nodes["repository:GitRepository:42"]
    assert node["node_type"] == "GitRepository"
    assert node["vcs"] == "gitlab"
    assert node["fullPath"] == "grp/demo"
    assert node["cloneUrl"] == "https://gl/grp/demo.git"
    assert node["defaultBranch"] == "main"
    assert node["externalToolId"] == "42"
    # empty head_sha is dropped (None-filtered)
    assert "headSha" not in node


def test_ingest_repositories_falls_back_to_full_path_id():
    c = _FakeClient()
    ingest_repositories(
        [{"vcs": "github", "full_path": "owner/repo", "clone_url": "x"}],
        client=c,
        graph="__commons__",
    )
    assert "repository:GitRepository:owner/repo" in c.txn.nodes


def test_ingest_worktrees_maps_and_links_repo():
    c = _FakeClient()
    res = ingest_worktrees(
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
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    wt = c.txn.nodes["repository:Worktree:worktree://agent-utilities/feat-x"]
    assert wt["node_type"] == "Worktree"
    assert wt["branchName"] == "feat/x"
    assert wt["worktreeStatus"] == "active"
    assert wt["dirty"] is True
    assert wt["aheadCount"] == 3
    assert "repository:GitRepository:agent-utilities" in c.txn.nodes
    assert c.txn.edges == [
        (
            "repository:Worktree:worktree://agent-utilities/feat-x",
            "repository:GitRepository:agent-utilities",
            {"relationship": "worktreeOf"},
        )
    ]


def test_ingest_projects_maps_and_links_repo():
    c = _FakeClient()
    res = ingest_projects(
        [
            {
                "name": "gitlab-api",
                "path": "repository://gitlab-api",
                "class": "clean",
                "dirty": False,
                "ahead_origin": 0,
            }
        ],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    proj = c.txn.nodes["repository:Project:gitlab-api"]
    assert proj["node_type"] == "Project"
    assert proj["validationStatus"] == "clean"
    assert proj["projectPath"].endswith("gitlab-api")
    assert c.txn.edges == [
        (
            "repository:Project:gitlab-api",
            "repository:GitRepository:gitlab-api",
            {"relationship": "projectOfRepository"},
        )
    ]


def test_retired_structural_alias_is_rejected():
    with pytest.raises(NativeIngestError, match="canonical node_type"):
        ingest_entities([{"id": "a", "type": "GitRepository"}], client=_FakeClient())


def test_empty_native_ingest_is_rejected():
    with pytest.raises(NativeIngestError, match="at least one entity"):
        ingest_entities([], client=_FakeClient())
