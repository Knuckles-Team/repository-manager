"""Characterization tests for CXA-FL-REPOSITORYMANAGER-03.

``repository_manager.docs_readiness_rollout.load_fleet_manifest`` (CCN 52) had
only 2 existing direct tests before this lane
(``tests/test_docs_readiness_fleet_rollout.py::test_packaged_manifest_is_exact_75_and_wave_counts``
happy path, and ``test_manifest_rejects_cardinality_drift`` one error path).
That leaves ~20 of its ~21 ``raise RolloutError(...)`` sites completely
uncovered. This file adds one targeted negative test per uncovered raise
site, each mutating exactly one field of the packaged, otherwise-valid
manifest and asserting the exact expected ``RolloutError`` code.

Every case here was run GREEN against the unmodified function before the
refactor commit, and is re-run unmodified (byte-identical) after it -- this
file, together with the existing
``tests/test_docs_readiness_fleet_rollout.py``, is the full characterization
baseline for this function.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from repository_manager import docs_readiness_rollout as rollout


def _base_data() -> dict[str, Any]:
    source = rollout.default_manifest_path()
    data: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    return candidate


def _mutate_schema_version(data: dict[str, Any]) -> dict[str, Any]:
    data["schema_version"] = "docs-readiness-rollout/v2"
    return data


def _mutate_extra_key(data: dict[str, Any]) -> dict[str, Any]:
    data["unexpected_extra_key"] = 1
    return data


def _mutate_missing_key(data: dict[str, Any]) -> dict[str, Any]:
    del data["manifest_name"]
    return data


def _mutate_name_empty(data: dict[str, Any]) -> dict[str, Any]:
    data["manifest_name"] = ""
    return data


def _mutate_name_not_str(data: dict[str, Any]) -> dict[str, Any]:
    data["manifest_name"] = 1
    return data


def _mutate_source_authority(data: dict[str, Any]) -> dict[str, Any]:
    data["source_authority"] = "wrong authority"
    return data


def _mutate_surface_policy_not_dict(data: dict[str, Any]) -> dict[str, Any]:
    data["surface_policy"] = ["not", "a", "dict"]
    return data


def _mutate_surface_policy_value(data: dict[str, Any]) -> dict[str, Any]:
    data["surface_policy"]["generator"] = "wrong-generator/v1"
    return data


def _mutate_expected_count_wrong_type(data: dict[str, Any]) -> dict[str, Any]:
    data["expected_publishable_count"] = "75"
    return data


def _mutate_excluded_wrong_length(data: dict[str, Any]) -> dict[str, Any]:
    data["excluded"] = []
    return data


def _mutate_excluded_wrong_shape(data: dict[str, Any]) -> dict[str, Any]:
    data["excluded"] = [{"identity": "agent-packages/agents/tests"}]
    return data


def _mutate_excluded_wrong_identity(data: dict[str, Any]) -> dict[str, Any]:
    data["excluded"] = [
        {"identity": "agent-packages/agents/other", "reason": "test-fixture-not-publishable"}
    ]
    return data


def _mutate_waves_not_list(data: dict[str, Any]) -> dict[str, Any]:
    data["waves"] = {}
    return data


def _mutate_wave_row_bad_shape(data: dict[str, Any]) -> dict[str, Any]:
    data["waves"][0] = {"number": 1, "name": "foundations"}
    return data


def _mutate_wave_row_number_invalid(data: dict[str, Any]) -> dict[str, Any]:
    data["waves"][0]["number"] = 0
    return data


def _mutate_wave_row_name_pattern_invalid(data: dict[str, Any]) -> dict[str, Any]:
    data["waves"][0]["name"] = "Foundations!"
    return data


def _mutate_wave_order_invalid(data: dict[str, Any]) -> dict[str, Any]:
    data["waves"][0]["number"], data["waves"][1]["number"] = (
        data["waves"][1]["number"],
        data["waves"][0]["number"],
    )
    return data


def _mutate_project_row_bad_shape(data: dict[str, Any]) -> dict[str, Any]:
    data["projects"][0] = {"identity": data["projects"][0]["identity"]}
    return data


def _mutate_project_wave_out_of_range(data: dict[str, Any]) -> dict[str, Any]:
    data["projects"][0]["wave"] = len(data["waves"]) + 1
    return data


def _mutate_duplicate_project(data: dict[str, Any]) -> dict[str, Any]:
    data["projects"][1]["identity"] = data["projects"][0]["identity"]
    return data


def _mutate_projects_count_mismatch(data: dict[str, Any]) -> dict[str, Any]:
    data["projects"] = data["projects"][:-1]
    return data


def _mutate_project_order_invalid(data: dict[str, Any]) -> dict[str, Any]:
    data["projects"][0], data["projects"][1] = data["projects"][1], data["projects"][0]
    return data


def _mutate_wave_cardinality_mismatch(data: dict[str, Any]) -> dict[str, Any]:
    # Move one project into a different (still in-range) wave without
    # changing project count, order, or identity -- only the per-wave count.
    for project in data["projects"]:
        if project["wave"] == 1:
            project["wave"] = 2
            return data
    raise AssertionError("fixture drift: no wave-1 project found")


def _mutate_findings_wrong_shape(data: dict[str, Any]) -> dict[str, Any]:
    data["source_findings"] = {"missing_pages_workflows": []}
    return data


def _mutate_findings_wrong_count(data: dict[str, Any]) -> dict[str, Any]:
    data["source_findings"]["missing_pages_workflows"] = []
    return data


def _mutate_findings_duplicate(data: dict[str, Any]) -> dict[str, Any]:
    value = data["source_findings"]["missing_pages_workflows"][0]
    data["source_findings"]["missing_pages_workflows"] = [value, value]
    return data


CASES: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]], str]] = [
    ("schema-version", _mutate_schema_version, "manifest-schema-invalid"),
    ("extra-key", _mutate_extra_key, "manifest-schema-invalid"),
    ("missing-key", _mutate_missing_key, "manifest-schema-invalid"),
    ("name-empty", _mutate_name_empty, "manifest-name-invalid"),
    ("name-not-str", _mutate_name_not_str, "manifest-name-invalid"),
    ("source-authority", _mutate_source_authority, "manifest-source-authority-invalid"),
    ("policy-not-dict", _mutate_surface_policy_not_dict, "manifest-surface-policy-invalid"),
    ("policy-value", _mutate_surface_policy_value, "manifest-surface-policy-invalid"),
    ("count-wrong-type", _mutate_expected_count_wrong_type, "manifest-cardinality-invalid"),
    ("excluded-length", _mutate_excluded_wrong_length, "manifest-exclusions-invalid"),
    ("excluded-shape", _mutate_excluded_wrong_shape, "manifest-exclusions-invalid"),
    ("excluded-identity", _mutate_excluded_wrong_identity, "manifest-exclusions-invalid"),
    ("waves-not-list", _mutate_waves_not_list, "manifest-waves-invalid"),
    ("wave-row-shape", _mutate_wave_row_bad_shape, "manifest-waves-invalid"),
    ("wave-row-number", _mutate_wave_row_number_invalid, "manifest-waves-invalid"),
    ("wave-row-name", _mutate_wave_row_name_pattern_invalid, "manifest-waves-invalid"),
    ("wave-order", _mutate_wave_order_invalid, "manifest-wave-order-invalid"),
    ("project-row-shape", _mutate_project_row_bad_shape, "manifest-project-invalid"),
    ("project-wave-range", _mutate_project_wave_out_of_range, "manifest-project-invalid"),
    ("duplicate-project", _mutate_duplicate_project, "manifest-duplicate-project"),
    ("projects-count", _mutate_projects_count_mismatch, "manifest-cardinality-invalid"),
    ("project-order", _mutate_project_order_invalid, "manifest-project-order-invalid"),
    ("wave-cardinality", _mutate_wave_cardinality_mismatch, "manifest-wave-cardinality-invalid"),
    ("findings-shape", _mutate_findings_wrong_shape, "manifest-findings-invalid"),
    ("findings-count", _mutate_findings_wrong_count, "manifest-findings-invalid"),
    ("findings-duplicate", _mutate_findings_duplicate, "manifest-findings-invalid"),
]


@pytest.mark.parametrize("name,mutate,expected_code", CASES, ids=[c[0] for c in CASES])
def test_load_fleet_manifest_rejects_each_mutated_field(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], dict[str, Any]],
    expected_code: str,
) -> None:
    del name
    data = mutate(_base_data())
    candidate = _write(tmp_path, data)
    with pytest.raises(rollout.RolloutError, match=expected_code):
        rollout.load_fleet_manifest(candidate)


def test_load_fleet_manifest_unmutated_base_data_still_loads(tmp_path: Path) -> None:
    """Proves ``_base_data()`` itself is a valid manifest, so every failure
    above is caused by exactly the one field each mutator changed."""

    candidate = _write(tmp_path, _base_data())
    manifest = rollout.load_fleet_manifest(candidate)
    assert manifest.expected_publishable_count == 75
