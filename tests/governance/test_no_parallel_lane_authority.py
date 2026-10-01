"""RM-GOVERNANCE-R001: no parallel Git-governance implementation outside RM.

repository-manager used to reach ``agent_utilities.governance.lanes`` for lane
arbitration (the shared-resource arbitration classes every concurrent lane on
a host depends on: partition, lease, append-only fragment, read-only). An
operator ruling moved that module here as ``repository_manager.governance.lanes``, hosted and
versioned as this package's own governance implementation rather than an
external dependency's. This test is the source-and-import proof the
requirement's verification column calls for: a static AST scan confirms no
production module under ``repository_manager/`` still imports the retired
external module, and an import test confirms the local module is this
package's live implementation instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_MANAGER_ROOT = Path(__file__).resolve().parents[2] / "repository_manager"

#: ``agent_utilities.governance.concept_reservation`` is a separate, still-out
#: -of-scope authority (RM-GOVERNANCE-07): ``concept_coordination/client.py``
#: names it only as a string a caller may inject, never imports it, and this
#: test does not pin that unrelated requirement.
RETIRED_LANE_MODULE = "agent_utilities.governance.lanes"
RETIRED_LANE_PACKAGE = "agent_utilities.governance"


def _production_python_files() -> list[Path]:
    return sorted(
        path
        for path in REPOSITORY_MANAGER_ROOT.rglob("*.py")
        if "governance" not in path.relative_to(REPOSITORY_MANAGER_ROOT).parts
    )


def _retired_import_from_hit(node: ast.ImportFrom) -> str | None:
    module = node.module or ""
    if module == RETIRED_LANE_MODULE:
        return f"from {module} import ..."
    if module == RETIRED_LANE_PACKAGE and any(
        alias.name == "lanes" for alias in node.names
    ):
        return f"from {module} import lanes"
    return None


def _retired_import_hit(node: ast.Import) -> str | None:
    retired_names = (RETIRED_LANE_MODULE, f"{RETIRED_LANE_PACKAGE}.lanes")
    return next(
        (f"import {alias.name}" for alias in node.names if alias.name in retired_names),
        None,
    )


def _imports_retired_lane_module(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            hit = _retired_import_from_hit(node)
        elif isinstance(node, ast.Import):
            hit = _retired_import_hit(node)
        else:
            hit = None
        if hit is not None:
            return hit
    return None


@pytest.mark.parametrize("path", _production_python_files(), ids=lambda p: str(p))
def test_no_production_module_imports_the_retired_lane_module(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hit = _imports_retired_lane_module(tree)
    assert hit is None, f"{path} still imports the retired module ({hit})"


def test_the_packages_own_lane_module_is_the_live_implementation() -> None:
    from repository_manager.governance import lanes

    # A representative surface used fleet-wide (merge_queue.py, safe_commit.py,
    # task_queue.py, build_queue.py, ...): present and owned by this package.
    assert lanes.__name__ == "repository_manager.governance.lanes"
    assert callable(lanes.hold_lease)
    assert callable(lanes.lane_scope)
    assert hasattr(lanes, "FragmentStore")


def test_concept_reservation_module_is_hosted_as_this_packages_own() -> None:
    from repository_manager.governance import concept_reservation

    assert concept_reservation.__name__ == (
        "repository_manager.governance.concept_reservation"
    )
