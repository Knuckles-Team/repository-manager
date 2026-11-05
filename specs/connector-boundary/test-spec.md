# RM-CONNECTOR — Test specification

Proof level: package import and server start with fake MCP and SDK clients and temporary Git repositories; no hosted forge or live graph is required.

## Bound tests

- Test: `tests/test_mcp_registration.py::test_mcp_tools_registration` and `tests/test_adapter_extraction.py::test_registry_order_and_extension_seam_are_deterministic`. *(RM-CONNECTOR-R001)*
- Test: `tests/governance/test_no_parallel_lane_authority.py::test_no_production_module_imports_the_retired_lane_module` and `tests/test_mcp_registration.py::test_mcp_tools_registration`. *(RM-CONNECTOR-R003)*
- Test: `tests/governance/test_no_parallel_lane_authority.py::test_the_packages_own_lane_module_is_the_live_implementation`, `tests/governance/test_no_parallel_lane_authority.py::test_concept_reservation_module_is_hosted_as_this_packages_own` and `tests/governance/test_no_parallel_lane_authority.py::test_no_production_module_imports_the_retired_lane_module`. *(RM-CONNECTOR-R004)*
- Test: `tests/test_adapter_extraction.py::test_per_operation_mcp_catalog_matches_contract` and `tests/test_skill_action_surface.py::test_skill_action_references_match_live_schema`. *(RM-CONNECTOR-R005)*
- Test: `tests/governance/test_no_parallel_lane_authority.py::test_no_production_module_imports_the_retired_lane_module`. The retired-name scan of the whole package, `tests/test_retired_names.py::test_no_retired_package_name_in_production_code` (planned), extends it beyond governance. *(RM-CONNECTOR-R006)*

## Planned

- `tests/test_import_inventory.py::test_every_orchestration_import_has_one_disposition_and_none_is_dynamic` (planned). Every import of the orchestration package has exactly one of the four classes and a dynamic import with a non-literal name fails. *(RM-CONNECTOR-R002)*
- `tests/test_retired_names.py::test_no_retired_package_name_in_production_code` (planned). A scan of every production module finds no retired package name. *(RM-CONNECTOR-R006)*

## Quality

Negative cases: an import cycle, a removed private symbol, a missing dependency, a stale lane fence and a red validation gate. Run the full repository pre-commit and the CCCC staged complexity check (no new function over cyclomatic 10 or cognitive 15), plus Ruff, mypy and pytest. jscpd and dupehound are not ordinary hooks of this repository; report a supplemental run or the tool gap. KISS means one implementation per responsibility and no transitional dual-write or fallback once cutover lands.
