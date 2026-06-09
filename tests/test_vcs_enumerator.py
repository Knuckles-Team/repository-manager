#!/usr/bin/python
"""Remote VCS enumeration (CONCEPT:KG-2.49).

Covers GitLab keyset pagination, GitHub page pagination, archived filtering, and
the max_repos cap — all offline via an injected fake HTTP client.
"""

from repository_manager.vcs_enumerator import (
    enumerate_github,
    enumerate_gitlab,
    write_manifest,
)


class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status

    def json(self):
        return self._p


class _FakeClient:
    """Returns queued responses in order; records the params it was called with."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def get(self, url, headers=None, params=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return self._pages.pop(0) if self._pages else _Resp([])

    def close(self):
        pass


def _gl_project(pid, name, archived=False):
    return {
        "id": pid,
        "path_with_namespace": f"group/{name}",
        "http_url_to_repo": f"https://gl/group/{name}.git",
        "web_url": f"https://gl/group/{name}",
        "default_branch": "main",
        "last_activity_at": "2026-01-01T00:00:00Z",
        "archived": archived,
    }


def test_gitlab_keyset_pagination_walks_all_pages():
    # Two full pages (100 each) then a short page → stop.
    page1 = _Resp([_gl_project(i, f"r{i}") for i in range(1, 101)])
    page2 = _Resp([_gl_project(i, f"r{i}") for i in range(101, 201)])
    page3 = _Resp([_gl_project(201, "r201")])
    client = _FakeClient([page1, page2, page3])
    refs = enumerate_gitlab(base_url="https://gl", token="t", client=client)
    assert len(refs) == 201
    assert refs[0]["vcs"] == "gitlab"
    assert refs[0]["full_path"] == "group/r1"
    # keyset cursor advanced by max id of each batch.
    assert client.calls[1]["params"]["id_after"] == 100
    assert client.calls[2]["params"]["id_after"] == 200


def test_gitlab_excludes_archived_by_default():
    page = _Resp([_gl_project(1, "live"), _gl_project(2, "old", archived=True)])
    refs = enumerate_gitlab(base_url="https://gl", token="t", client=_FakeClient([page]))
    assert [r["full_path"] for r in refs] == ["group/live"]


def test_gitlab_max_repos_cap():
    page = _Resp([_gl_project(i, f"r{i}") for i in range(1, 101)])
    refs = enumerate_gitlab(
        base_url="https://gl", token="t", client=_FakeClient([page]), max_repos=10
    )
    assert len(refs) == 10


def test_gitlab_per_group_sets_include_subgroups():
    page = _Resp([_gl_project(1, "a")])
    client = _FakeClient([page])
    enumerate_gitlab(base_url="https://gl", token="t", groups=["42"], client=client)
    assert client.calls[0]["url"].endswith("/groups/42/projects")
    assert client.calls[0]["params"]["include_subgroups"] == "true"


def _gh_repo(name, archived=False):
    return {
        "full_name": f"org/{name}",
        "clone_url": f"https://github.com/org/{name}.git",
        "html_url": f"https://github.com/org/{name}",
        "default_branch": "main",
        "pushed_at": "2026-01-01T00:00:00Z",
        "archived": archived,
    }


def test_github_page_pagination():
    page1 = _Resp([_gh_repo(f"r{i}") for i in range(100)])
    page2 = _Resp([_gh_repo("last")])
    client = _FakeClient([page1, page2])
    refs = enumerate_github(token="t", orgs=["org"], client=client)
    assert len(refs) == 101
    assert refs[-1]["full_path"] == "org/last"
    assert client.calls[1]["params"]["page"] == 2


def test_write_manifest(tmp_path):
    refs = [_github_ref_dict()]
    path = write_manifest(refs, "run123", out_dir=str(tmp_path))
    assert path.endswith("vcs_enumeration_run123.json")
    import json

    data = json.loads(open(path).read())
    assert data["count"] == 1 and data["run_id"] == "run123"


def _github_ref_dict():
    return {"vcs": "github", "full_path": "org/x", "clone_path": "/c/x", "head_sha": ""}
