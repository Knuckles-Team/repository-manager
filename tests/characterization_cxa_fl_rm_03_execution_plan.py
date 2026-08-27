"""Characterization test for CXA-FL-REPOSITORYMANAGER-03.

Pins one branch of ``BuildWorker._execution_plan``
(``repository_manager/build_worker.py``) that the existing
``tests/test_durable_build_broker.py`` suite (63 tests) does not reach:
the ``payload.base_sha != view.base_sha`` identity check.

Mutation-proof performed manually before writing this file (mutation
reverted before commit): deleting that check's `if`/`raise` block left the
full existing ``test_durable_build_broker.py`` suite green (63/63) -- i.e.
no existing test drives a WorkItem view whose ``base_sha`` disagrees with
its persisted execution payload. This file closes that gap so the refactor
commit (extracting the identity checks into
``_execution_plan_validate_workitem_identity``) cannot silently drop or
reorder this check.

This file is additive only; ``tests/test_durable_build_broker.py`` remains
the primary characterization baseline and is run before and after the
refactor, unmodified, with an identical (63 passed) result required.

A second gap was found the same way and is NOT closed here: the
``payload.degraded_reason == "dirty-tree"`` check inside ``_execution_plan``
(the ``dirty`` branch) is also untested by the existing suite -- but for a
structural reason, not an oversight: ``test_dirty_canonical_build_is_refused_
before_durable_submission`` proves the public ``BuildService.submit`` path
already refuses a dirty tree before a WorkItem is ever durably created, so
no fixture in the existing suite can produce a persisted payload with
``degraded_reason == "dirty-tree"`` to feed back through ``_execution_plan``.
Reaching it would require hand-constructing a typed execution payload below
the public API and bypassing several `pydantic` field validators on
``ExecutionPayload`` (see ``repository_manager/development/payloads.py``),
which is out of scope for a pure-refactor characterization pass. Recorded in
the lane report as a named, deliberately-uncovered branch (defense-in-depth,
not dead code -- it guards against a corrupted/tampered persisted payload,
not against anything reachable via the public submission path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest

from repository_manager.build_service import BuildService
from repository_manager.build_worker import BuildAuthority, BuildWorker, BuildWorkerError
from repository_manager.development.jobs import FakeRepositoryJobPort, RepositoryJobService
from repository_manager.development import JobAuthorization


def _repo(tmp_path: Path) -> Path:
    """Same fixture shape as test_durable_build_broker.py's ``_repo`` helper
    (inlined here rather than imported, since pytest does not put sibling
    test modules on ``sys.path`` by module name in this repo's layout)."""

    repo = tmp_path / "same-basename"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "RMDD tests"], cwd=repo, check=True)
    (repo / "build_script.py").write_text("print(1)\n", encoding="utf-8")
    (repo / ".buildcache.yaml").write_text(
        """schema_version: 2
base: main
specs:
  - name: test-build
    command: [python3, build_script.py]
    artifacts: [out.txt]
    resource_class: light-check
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    return repo


class _TypedJobService:
    """Same shape as test_durable_build_broker.py's ``_TypedJobService``
    fake (inlined for the same reason as ``_repo`` above)."""

    def __init__(self) -> None:
        self.port = FakeRepositoryJobPort()
        self.service = RepositoryJobService(self.port)

    def submit(self, *args: object, **kwargs: object) -> object:
        return self.service.submit(*args, **kwargs)  # type: ignore[arg-type]

    def exact(self, job_id: str, *, owner_id: str = "repository-manager") -> object:
        return self.service.get_exact_execution_input(
            job_id,
            auth=JobAuthorization(tenant_id="repository-manager", owner_id=owner_id),
        )


def test_execution_plan_rejects_base_sha_disagreeing_with_persisted_payload(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    typed = _TypedJobService()
    submitted = BuildService(typed).submit(repo_path=repo, spec_name="test-build")
    view = typed.port.rows[submitted["job_id"]]

    class Authority:
        def get_exact_execution_input(self, job_id: str) -> object:
            return typed.exact(job_id, owner_id=view.owner_id)

        def execution_input_authority_available(self) -> bool:
            return True

        def claim(self, job_id: str, *, token: str) -> dict[str, object]:
            del token
            return {
                "job_id": job_id,
                "work_item_id": view.work_item_id,
                "attempt": 1,
                "fence": "f1",
            }

        def is_current(self, job_id: str, claim: object) -> bool:
            del job_id
            return isinstance(claim, dict) and claim.get("fence") == "f1"

    worker = BuildWorker(cast(BuildAuthority, Authority()), None)
    claim = worker.authority.claim(view.job_id, token="test")

    # A different (but still well-formed) base SHA than what was persisted
    # in the exact execution payload at submission time.
    flipped = ("0" if view.base_sha[0] != "0" else "1") + view.base_sha[1:]
    forged_view = view.model_copy(update={"base_sha": flipped})

    with pytest.raises(
        BuildWorkerError, match="build payload SHA disagrees with WorkItem authority"
    ):
        worker._execution_plan(  # noqa: SLF001
            forged_view, repo_path=repo, spec_name="test-build", claim=claim
        )
