"""Native epistemic-graph typed-node ingestion — Wire-First coverage.

Exercises the real ``ingest_entities`` / ``ingest_repositories`` / ``ingest_worktrees`` /
``ingest_projects`` seams with a fake engine client (no engine required), asserting the txn
add_node/commit + edge calls and the record → :GitRepository / :Worktree / :Project mappings.
CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from repository_manager.kg_ingest import (
    ingest_entities,
    ingest_projects,
    ingest_repositories,
    ingest_worktrees,
)


class _FakeTxn:
    def __init__(self):
        self.nodes = {}
        self.committed = False

    def begin(self, graph=None):
        self.graph = graph
        return "txn-1"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = props

    def commit(self, txn):
        self.committed = True
        return True


class _FakeEdges:
    def __init__(self):
        self.edges = []

    def add(self, src, dst, props):
        self.edges.append((src, dst, props))


class _FakeClient:
    def __init__(self):
        self.txn = _FakeTxn()
        self.edges = _FakeEdges()


def test_ingest_entities_writes_nodes_and_edges():
    c = _FakeClient()
    res = ingest_entities(
        [
            {"id": "a", "type": "GitRepository", "name": "p"},
            {"id": "b", "type": "Worktree"},
        ],
        [{"source": "b", "target": "a", "type": "worktreeOf"}],
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 2, "edges": 1}
    assert c.txn.committed is True
    assert set(c.txn.nodes) == {"a", "b"}
    # provenance is stamped
    assert c.txn.nodes["a"]["source"] == "repository-manager"
    assert c.txn.nodes["a"]["domain"] == "repository"
    assert c.edges.edges == [("b", "a", {"type": "worktreeOf"})]


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
    assert node["type"] == "GitRepository"
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
                "path": "/home/apps/worktrees/agent-utilities/feat-x",
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
    wt = c.txn.nodes["repository:Worktree:/home/apps/worktrees/agent-utilities/feat-x"]
    assert wt["type"] == "Worktree"
    assert wt["branchName"] == "feat/x"
    assert wt["worktreeStatus"] == "active"
    assert wt["dirty"] is True
    assert wt["aheadCount"] == 3
    assert "repository:GitRepository:agent-utilities" in c.txn.nodes
    assert c.edges.edges == [
        (
            "repository:Worktree:/home/apps/worktrees/agent-utilities/feat-x",
            "repository:GitRepository:agent-utilities",
            {"type": "worktreeOf"},
        )
    ]


def test_ingest_projects_maps_and_links_repo():
    c = _FakeClient()
    res = ingest_projects(
        [
            {
                "name": "gitlab-api",
                "path": "/home/apps/workspace/agent-packages/agents/gitlab-api",
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
    assert proj["type"] == "Project"
    assert proj["validationStatus"] == "clean"
    assert proj["projectPath"].endswith("gitlab-api")
    assert c.edges.edges == [
        (
            "repository:Project:gitlab-api",
            "repository:GitRepository:gitlab-api",
            {"type": "projectOfRepository"},
        )
    ]


def test_ingest_noops_without_engine():
    # No injected client + no reachable engine -> clean no-op.
    assert ingest_entities([{"id": "a", "type": "GitRepository"}]) is None


def test_ingest_empty_is_noop():
    assert ingest_entities([], client=_FakeClient()) is None
    assert ingest_repositories([], client=_FakeClient()) is None
    assert ingest_worktrees([], client=_FakeClient()) is None
    assert ingest_projects([], client=_FakeClient()) is None
