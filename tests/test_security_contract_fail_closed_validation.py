"""Characterization tests for security_contract.py.

security_contract.py is a security gate (bounded SPDX license-policy
validation + sandboxed hook execution). Per the wave's SPECIAL CASE rule for
gate/scanner files, these tests plant known-bad inputs the gate is supposed
to reject (an unsupported contract version, a denied license, a hook that
reports a crash, an evidence path that escapes the result root, ...) and
confirm each fails closed, alongside the happy-path "this exact well-formed
input is accepted" cases. Run unmodified before and after the extract-method
refactor of load_contract/run_hook/check_licenses/_tokenize_spdx/
_component_licenses/_validate_hook_evidence; results must be identical.

There is no pre-existing dedicated test module for this file (only one
resource-module-absent regression lives in test_import_safety_gate.py); this
file is the primary characterization baseline for the refactor.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "security_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "security_contract_char", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


def _valid_contract_dict() -> dict:
    return {
        "version": 2,
        "hooks": {
            "fuzz": {
                "argv": [sys.executable, "-c", "pass"],
                "timeout_seconds": 30,
                "evidence": "security-results/fuzz.json",
                "min_cases": 1,
            },
            "authenticated_negative": {
                "argv": [sys.executable, "-c", "pass"],
                "timeout_seconds": 30,
                "evidence": "security-results/authenticated-negative.json",
                "min_cases": 1,
            },
        },
        "license_policy": {
            "allowed": ["MIT", "Apache-2.0"],
            "allowed_exceptions": [],
            "denied": ["GPL-3.0-only"],
        },
    }


def _write_contract(root: Path, contract: dict, name: str = "security-contract.json") -> str:
    (root / name).write_text(json.dumps(contract), encoding="utf-8")
    return name


# ---------------------------------------------------------------------------
# _tokenize_spdx
# ---------------------------------------------------------------------------


def test_tokenize_spdx_single_identifier():
    assert sc._tokenize_spdx("MIT") == ("MIT",)


def test_tokenize_spdx_and_expression():
    assert sc._tokenize_spdx("MIT AND Apache-2.0") == ("MIT", "AND", "Apache-2.0")


def test_tokenize_spdx_parens_and_with():
    assert sc._tokenize_spdx("(MIT WITH Classpath-exception-2.0)") == (
        "(",
        "MIT",
        "WITH",
        "Classpath-exception-2.0",
        ")",
    )


def test_tokenize_spdx_empty_string_rejected():
    with pytest.raises(sc._SpdxSyntaxError):
        sc._tokenize_spdx("")


def test_tokenize_spdx_embedded_nul_rejected():
    with pytest.raises(sc._SpdxSyntaxError):
        sc._tokenize_spdx("MIT\x00")


def test_tokenize_spdx_invalid_identifier_rejected():
    with pytest.raises(sc._SpdxSyntaxError):
        sc._tokenize_spdx("MIT$BAD")


def test_tokenize_spdx_non_string_rejected():
    with pytest.raises(sc._SpdxSyntaxError):
        sc._tokenize_spdx(None)  # type: ignore[arg-type]


def test_tokenize_spdx_too_many_tokens_rejected():
    expression = " AND ".join(["MIT"] * 300)
    with pytest.raises(sc._SpdxSyntaxError):
        sc._tokenize_spdx(expression)


# ---------------------------------------------------------------------------
# load_contract
# ---------------------------------------------------------------------------


def test_load_contract_accepts_well_formed_contract(tmp_path):
    name = _write_contract(tmp_path, _valid_contract_dict())
    loaded = sc.load_contract(tmp_path, name)
    assert loaded["version"] == 2
    assert set(loaded["hooks"]) == {"fuzz", "authenticated_negative"}


def test_load_contract_rejects_unsupported_version(tmp_path):
    contract = _valid_contract_dict()
    contract["version"] = 1
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="version is unsupported"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_missing_hook_kind(tmp_path):
    contract = _valid_contract_dict()
    del contract["hooks"]["fuzz"]
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="must declare every hook"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_empty_argv(tmp_path):
    contract = _valid_contract_dict()
    contract["hooks"]["fuzz"]["argv"] = []
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="argv is invalid"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_out_of_range_timeout(tmp_path):
    contract = _valid_contract_dict()
    contract["hooks"]["fuzz"]["timeout_seconds"] = 0
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="timeout is invalid"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_absolute_evidence_path(tmp_path):
    contract = _valid_contract_dict()
    contract["hooks"]["fuzz"]["evidence"] = "/etc/passwd"
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="evidence path is invalid"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_empty_allow_list(tmp_path):
    contract = _valid_contract_dict()
    contract["license_policy"]["allowed"] = []
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="must declare an allow-list"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_overlapping_allow_and_deny(tmp_path):
    contract = _valid_contract_dict()
    contract["license_policy"]["denied"] = ["MIT"]
    name = _write_contract(tmp_path, contract)
    with pytest.raises(sc.SecurityContractError, match="allow and deny lists overlap"):
        sc.load_contract(tmp_path, name)


def test_load_contract_rejects_path_escaping_root(tmp_path):
    with pytest.raises(sc.SecurityContractError, match="path is invalid"):
        sc.load_contract(tmp_path, "../outside.json")


# ---------------------------------------------------------------------------
# _component_licenses
# ---------------------------------------------------------------------------


def test_component_licenses_expression_form():
    component = {"licenses": [{"expression": "MIT OR Apache-2.0"}]}
    values, malformed = sc._component_licenses(component)
    assert values == ("MIT OR Apache-2.0",)
    assert malformed is False


def test_component_licenses_id_form():
    component = {"licenses": [{"license": {"id": "MIT"}}]}
    values, malformed = sc._component_licenses(component)
    assert values == ("MIT",)
    assert malformed is False


def test_component_licenses_no_licenses_is_not_malformed():
    assert sc._component_licenses({}) == ((), False)


def test_component_licenses_malformed_declaration():
    component: dict[str, object] = {"licenses": [{"license": {}}]}
    values, malformed = sc._component_licenses(component)
    assert values == ()
    assert malformed is True


def test_component_licenses_too_many_declarations_is_malformed():
    component = {"licenses": [{"expression": "MIT"}] * 100}
    values, malformed = sc._component_licenses(component)
    assert values == ()
    assert malformed is True


# ---------------------------------------------------------------------------
# check_licenses
# ---------------------------------------------------------------------------


def _sbom(components: list[dict]) -> dict:
    return {"bomFormat": "CycloneDX", "components": components}


def test_check_licenses_accepts_only_allowed_licenses(tmp_path):
    contract = sc.load_contract(tmp_path, _write_contract(tmp_path, _valid_contract_dict()))
    sbom_name = "sbom.json"
    (tmp_path / sbom_name).write_text(
        json.dumps(_sbom([{"licenses": [{"expression": "MIT"}]}])), encoding="utf-8"
    )
    sc.check_licenses(tmp_path, contract, sbom_name, "security-results/licenses.json")
    output = json.loads((tmp_path / "security-results" / "licenses.json").read_text())
    assert output["passed"] is True
    assert output["violations"] == 0


def test_check_licenses_rejects_denied_license(tmp_path):
    contract = sc.load_contract(tmp_path, _write_contract(tmp_path, _valid_contract_dict()))
    sbom_name = "sbom.json"
    (tmp_path / sbom_name).write_text(
        json.dumps(_sbom([{"licenses": [{"expression": "GPL-3.0-only"}]}])),
        encoding="utf-8",
    )
    with pytest.raises(sc.SecurityContractError, match="violates license policy"):
        sc.check_licenses(tmp_path, contract, sbom_name, "security-results/licenses.json")
    output = json.loads((tmp_path / "security-results" / "licenses.json").read_text())
    assert output["passed"] is False
    assert output["violations"] == 1


def test_check_licenses_rejects_non_cyclonedx_format(tmp_path):
    contract = sc.load_contract(tmp_path, _write_contract(tmp_path, _valid_contract_dict()))
    sbom_name = "sbom.json"
    (tmp_path / sbom_name).write_text(json.dumps({"bomFormat": "SPDX"}), encoding="utf-8")
    with pytest.raises(sc.SecurityContractError, match="not CycloneDX"):
        sc.check_licenses(tmp_path, contract, sbom_name, "security-results/licenses.json")


def test_check_licenses_treats_unknown_license_as_failing(tmp_path):
    contract = sc.load_contract(tmp_path, _write_contract(tmp_path, _valid_contract_dict()))
    sbom_name = "sbom.json"
    (tmp_path / sbom_name).write_text(json.dumps(_sbom([{}])), encoding="utf-8")
    with pytest.raises(sc.SecurityContractError, match="violates license policy"):
        sc.check_licenses(tmp_path, contract, sbom_name, "security-results/licenses.json")
    output = json.loads((tmp_path / "security-results" / "licenses.json").read_text())
    assert output["unknown"] == 1
    assert output["passed"] is False


# ---------------------------------------------------------------------------
# _validate_hook_evidence
# ---------------------------------------------------------------------------


def _hook(min_cases: int = 1) -> dict:
    return {
        "argv": [sys.executable, "-c", "pass"],
        "timeout_seconds": 30,
        "evidence": "security-results/fuzz.json",
        "min_cases": min_cases,
    }


def _evidence(**overrides) -> dict:
    base = {
        "version": 1,
        "kind": "fuzz",
        "passed": True,
        "cases": 5,
        "failures": 0,
        "crashes": 0,
    }
    base.update(overrides)
    return base


def test_validate_hook_evidence_accepts_passing_evidence():
    sc._validate_hook_evidence("fuzz", _hook(), _evidence())


def test_validate_hook_evidence_rejects_missing_field():
    evidence = _evidence()
    del evidence["cases"]
    with pytest.raises(sc.SecurityContractError, match="schema is invalid"):
        sc._validate_hook_evidence("fuzz", _hook(), evidence)


def test_validate_hook_evidence_rejects_not_passed():
    with pytest.raises(sc.SecurityContractError, match="did not pass"):
        sc._validate_hook_evidence("fuzz", _hook(), _evidence(passed=False))


def test_validate_hook_evidence_rejects_below_min_cases():
    with pytest.raises(sc.SecurityContractError, match="threshold was not met"):
        sc._validate_hook_evidence("fuzz", _hook(min_cases=100), _evidence(cases=5))


def test_validate_hook_evidence_rejects_fuzz_crash():
    with pytest.raises(sc.SecurityContractError, match="reported a crash"):
        sc._validate_hook_evidence("fuzz", _hook(), _evidence(crashes=1))


def test_validate_hook_evidence_rejects_authenticated_bypass():
    evidence = _evidence(kind="authenticated_negative", unauthorized_acceptances=1)
    with pytest.raises(sc.SecurityContractError, match="reported a bypass"):
        sc._validate_hook_evidence("authenticated_negative", _hook(), evidence)


# ---------------------------------------------------------------------------
# run_hook (real subprocess execution)
# ---------------------------------------------------------------------------


def _hook_contract(argv: list[str]) -> dict:
    contract = _valid_contract_dict()
    contract["hooks"]["fuzz"]["argv"] = argv
    return contract


def test_run_hook_accepts_a_passing_hook(tmp_path):
    evidence_path = tmp_path / "security-results" / "fuzz.json"
    script = (
        "import json, pathlib; "
        f"p = pathlib.Path({str(evidence_path)!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(json.dumps({'version': 1, 'kind': 'fuzz', 'passed': True, "
        "'cases': 10, 'failures': 0, 'crashes': 0}))"
    )
    contract = _hook_contract([sys.executable, "-c", script])
    sc.run_hook(tmp_path, contract, "fuzz", "security-results")


def test_run_hook_rejects_nonzero_exit(tmp_path):
    contract = _hook_contract([sys.executable, "-c", "import sys; sys.exit(1)"])
    with pytest.raises(sc.SecurityContractError, match="returned a failure"):
        sc.run_hook(tmp_path, contract, "fuzz", "security-results")


def test_run_hook_rejects_missing_evidence(tmp_path):
    contract = _hook_contract([sys.executable, "-c", "pass"])
    with pytest.raises(sc.SecurityContractError):
        sc.run_hook(tmp_path, contract, "fuzz", "security-results")


def test_run_hook_rejects_evidence_reporting_a_crash(tmp_path):
    evidence_path = tmp_path / "security-results" / "fuzz.json"
    script = (
        "import json, pathlib; "
        f"p = pathlib.Path({str(evidence_path)!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(json.dumps({'version': 1, 'kind': 'fuzz', 'passed': True, "
        "'cases': 10, 'failures': 0, 'crashes': 1}))"
    )
    contract = _hook_contract([sys.executable, "-c", script])
    with pytest.raises(sc.SecurityContractError, match="reported a crash"):
        sc.run_hook(tmp_path, contract, "fuzz", "security-results")


def test_run_hook_rejects_invalid_kind(tmp_path):
    contract = _hook_contract([sys.executable, "-c", "pass"])
    with pytest.raises(sc.SecurityContractError, match="kind is invalid"):
        sc.run_hook(tmp_path, contract, "not-a-real-kind", "security-results")


def test_run_hook_rejects_evidence_path_outside_result_root(tmp_path):
    contract = _hook_contract([sys.executable, "-c", "pass"])
    contract["hooks"]["fuzz"]["evidence"] = "elsewhere/fuzz.json"
    with pytest.raises(
        sc.SecurityContractError, match="must stay in the result root"
    ):
        sc.run_hook(tmp_path, contract, "fuzz", "security-results")


def test_run_hook_fails_closed_without_unix_resource_support(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_WINDOWS", False)
    monkeypatch.setattr(sc, "_resource", None)
    contract = _hook_contract([sys.executable, "-c", "pass"])
    with pytest.raises(sc.SecurityContractError, match="Unix resource"):
        sc.run_hook(tmp_path, contract, "fuzz", "security-results")
