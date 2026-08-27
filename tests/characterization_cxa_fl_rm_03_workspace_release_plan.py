"""Characterization tests for CXA-FL-REPOSITORYMANAGER-03.

Pins 5 branches of ``WorkspaceReleasePlan.__post_init__``
(``repository_manager/development/workspace_release.py``, CCN 52) that the
existing ``tests/test_workspace_release_dag.py`` (729 lines) and
``tests/test_workspace_release_plan.py`` (1491 lines) -- 116 tests combined
-- do not reach.

Each gap below was confirmed by mutation testing against the UNMODIFIED
function before this lane touched it (mutation applied, full 116-test
existing suite run and stayed green, mutation reverted):

- deleting the stage self-dependency check
  (``if stage.stage_id in stage.depends_on: raise ...``)
- deleting the unknown-stage-dependency check
  (``if set(stage.depends_on) - stage_id_set: raise ...``)
- deleting the plan-level push-without-consent check
  (``if stage.stage == ReleaseStage.PUSH and not self.allow_push: raise
  ...`` -- distinct from ``PlanStage.__post_init__``'s own push/consent
  gate, which IS covered by
  ``test_frozen_c11_plan_has_stable_digest_and_no_push_without_consent``)
- deleting the parallel-group duplicate-project-within-a-group check
  (``if len(normalized) != len(set(normalized)): raise ...``)
- (found while auditing the digest branch, not by a failed mutation) no
  existing test constructs a plan with an explicit, deliberately-wrong
  ``plan_digest`` to exercise the "does not match frozen contents" branch;
  every existing test either omits ``plan_digest`` (auto-computed) or
  round-trips a plan's own correct digest.

Each new test below reproduces its gap: green against the unmodified
function, and (verified manually, reverted before commit) fails when its
target check is deleted.

This file is additive only; the existing 116-test suite remains the primary
characterization baseline and is run before and after the refactor,
unmodified, with an identical (116 passed) result required.
"""

from __future__ import annotations

import pytest

from repository_manager.development.workspace_release import (
    DependencyEdge,
    Ecosystem,
    PackageKey,
    PackageRecord,
    PlanStage,
    ProjectRecord,
    ReleaseStage,
    Version,
    VersionSource,
    WorkspaceReleaseError,
    WorkspaceReleasePlan,
    plan_digest,
)


def _package(repository: str, name: str) -> PackageRecord:
    key = PackageKey(repository, Ecosystem.PYTHON, name)
    version = Version("1.0.0")
    return PackageRecord(
        key=key,
        version=version,
        version_sources=(VersionSource("fixture", version),),
        dependencies=(),
    )


def _project(repository: str) -> ProjectRecord:
    return ProjectRecord(repository_id=repository, packages=(_package(repository, "pkg"),))


def _plan(**overrides: object) -> WorkspaceReleasePlan:
    project = _project("packages/demo")
    defaults: dict[str, object] = dict(
        workspace_id="workspace:test",
        source_sha="c" * 40,
        selected_projects=("packages/demo",),
        projects=(project,),
        stages=(
            PlanStage(
                stage_id="validate:repo:packages/demo",
                stage=ReleaseStage.VALIDATE,
                project_id="packages/demo",
            ),
        ),
        parallel_groups=(("packages/demo",),),
    )
    defaults.update(overrides)
    return WorkspaceReleasePlan(**defaults)  # type: ignore[arg-type]


def test_stage_cannot_depend_on_itself() -> None:
    stage = PlanStage(
        stage_id="validate:repo:packages/demo",
        stage=ReleaseStage.VALIDATE,
        project_id="packages/demo",
        depends_on=("validate:repo:packages/demo",),
    )
    with pytest.raises(WorkspaceReleaseError, match="cannot depend on itself"):
        _plan(stages=(stage,))


def test_stage_cannot_depend_on_an_unknown_stage() -> None:
    stage = PlanStage(
        stage_id="validate:repo:packages/demo",
        stage=ReleaseStage.VALIDATE,
        project_id="packages/demo",
        depends_on=("build:repo:packages/demo",),
    )
    with pytest.raises(WorkspaceReleaseError, match="depends on an unknown stage"):
        _plan(stages=(stage,))


def test_push_stage_requires_plan_level_push_consent() -> None:
    """Distinct from ``PlanStage.__post_init__``'s own push/consent gate:
    a PUSH ``PlanStage`` can be constructed on its own with
    ``requires_consent=True``, but ``WorkspaceReleasePlan.__post_init__``
    additionally requires the *plan*'s ``allow_push`` to be set."""

    push_stage = PlanStage(
        stage_id="push:repo:packages/demo",
        stage=ReleaseStage.PUSH,
        project_id="packages/demo",
        requires_consent=True,
    )
    with pytest.raises(WorkspaceReleaseError, match="requires plan push consent"):
        _plan(stages=(push_stage,), allow_push=False)

    # Sanity: the same stage succeeds once plan-level consent is granted.
    plan = _plan(stages=(push_stage,), allow_push=True)
    assert plan.allow_push is True


def test_parallel_group_cannot_duplicate_a_project() -> None:
    with pytest.raises(WorkspaceReleaseError, match="must not duplicate projects"):
        _plan(parallel_groups=(("packages/demo", "packages/demo"),))


def test_explicit_plan_digest_must_match_frozen_contents() -> None:
    baseline = _plan()
    correct = baseline.plan_digest
    wrong = ("0" if correct[0] != "0" else "1") + correct[1:]
    with pytest.raises(WorkspaceReleaseError, match="does not match frozen contents"):
        _plan(plan_digest=wrong)

    # Sanity: the correct digest round-trips (already implied by other
    # existing tests, repeated here so this file stands on its own).
    reloaded = _plan(plan_digest=correct)
    assert reloaded.plan_digest == plan_digest(reloaded)
