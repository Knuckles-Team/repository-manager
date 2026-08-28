"""Characterization tests for check_import_safety.py internals.

Pins the branch outputs of ``_windows_condition``, ``_windows_platform_value``,
and the ``_WindowsImportVisitor._visit`` dispatch before an extract-method
refactor splits each into smaller per-node-type helpers. These are exercised
only indirectly (through ``--simulate-windows`` end-to-end subprocess runs) by
the pre-existing ``tests/test_import_safety_gate.py`` suite, which does not
pin every AST-shape branch individually. Both suites run unmodified before and
after the refactor and must produce identical results.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_import_safety.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_import_safety_char", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


# ---------------------------------------------------------------------------
# _windows_platform_value
# ---------------------------------------------------------------------------


def test_platform_value_sys_platform_attribute():
    assert gate._windows_platform_value(_expr("sys.platform")) == "win32"


def test_platform_value_os_name_attribute():
    assert gate._windows_platform_value(_expr("os.name")) == "nt"


def test_platform_value_platform_system_call():
    assert gate._windows_platform_value(_expr("platform.system()")) == "Windows"


def test_platform_value_platform_system_call_with_args_is_none():
    assert gate._windows_platform_value(_expr("platform.system(1)")) is None


def test_platform_value_unrelated_attribute_is_none():
    assert gate._windows_platform_value(_expr("foo.bar")) is None


def test_platform_value_non_attribute_non_call_is_none():
    assert gate._windows_platform_value(_expr("1")) is None


# ---------------------------------------------------------------------------
# _windows_condition
# ---------------------------------------------------------------------------


def test_condition_bool_constant_true():
    assert gate._windows_condition(_expr("True")) is True


def test_condition_bool_constant_false():
    assert gate._windows_condition(_expr("False")) is False


def test_condition_type_checking_name_is_false():
    assert gate._windows_condition(_expr("TYPE_CHECKING")) is False


def test_condition_not_of_known_value():
    assert gate._windows_condition(_expr("not True")) is False
    assert gate._windows_condition(_expr("not False")) is True


def test_condition_not_of_unknown_value_is_none():
    assert gate._windows_condition(_expr("not x")) is None


def test_condition_and_all_true():
    assert gate._windows_condition(_expr("True and True")) is True


def test_condition_and_any_false_short_circuits_false():
    assert gate._windows_condition(_expr("False and x")) is False


def test_condition_and_unknown_is_none():
    assert gate._windows_condition(_expr("True and x")) is None


def test_condition_or_any_true():
    assert gate._windows_condition(_expr("False or True")) is True


def test_condition_or_all_false():
    assert gate._windows_condition(_expr("False or False")) is False


def test_condition_or_unknown_is_none():
    assert gate._windows_condition(_expr("False or x")) is None


def test_condition_startswith_true():
    assert gate._windows_condition(_expr("sys.platform.startswith('win')")) is True


def test_condition_startswith_false():
    assert gate._windows_condition(_expr("sys.platform.startswith('lin')")) is False


def test_condition_endswith_true():
    assert gate._windows_condition(_expr("os.name.endswith('t')")) is True


def test_condition_startswith_unknown_platform_probe_is_none():
    assert gate._windows_condition(_expr("foo.startswith('win')")) is None


def test_condition_startswith_two_args_is_none():
    assert gate._windows_condition(_expr("sys.platform.startswith('win', 1)")) is None


def test_condition_compare_eq_true():
    assert gate._windows_condition(_expr("sys.platform == 'win32'")) is True


def test_condition_compare_eq_false():
    assert gate._windows_condition(_expr("sys.platform == 'linux'")) is False


def test_condition_compare_noteq_true():
    assert gate._windows_condition(_expr("sys.platform != 'linux'")) is True


def test_condition_compare_in_tuple_true():
    assert gate._windows_condition(_expr("sys.platform in ('win32', 'cygwin')")) is True


def test_condition_compare_notin_tuple_true():
    assert gate._windows_condition(_expr("sys.platform not in ('linux', 'darwin')")) is True


def test_condition_compare_chained_is_none():
    assert gate._windows_condition(_expr("1 < sys.platform < 3")) is None


def test_condition_compare_unrelated_left_is_none():
    assert gate._windows_condition(_expr("foo == 'win32'")) is None


def test_condition_compare_non_string_comparator_is_none():
    assert gate._windows_condition(_expr("sys.platform == 1")) is None


def test_condition_other_node_shape_is_none():
    assert gate._windows_condition(_expr("[1, 2]")) is None


# ---------------------------------------------------------------------------
# _WindowsImportVisitor / _visit
# ---------------------------------------------------------------------------


def _scan(source: str) -> list[str]:
    tree = ast.parse(source)
    return gate._WindowsImportVisitor(Path("x.py")).scan(tree)


def test_visit_unguarded_posix_import_is_flagged():
    findings = _scan("import fcntl\n")
    assert len(findings) == 1
    assert "fcntl" in findings[0]


def test_visit_guarded_by_platform_false_branch_is_not_flagged():
    findings = _scan("import sys\nif sys.platform != 'win32':\n    import fcntl\n")
    assert findings == []


def test_visit_guarded_by_import_error_try_is_not_flagged():
    findings = _scan("try:\n    import fcntl\nexcept ImportError:\n    fcntl = None\n")
    assert findings == []


def test_visit_import_inside_true_branch_still_flagged():
    findings = _scan("if True:\n    import fcntl\n")
    assert len(findings) == 1


def test_visit_from_import_posix_only_module_is_flagged():
    findings = _scan("from resource import getrlimit\n")
    assert len(findings) == 1
    assert "resource" in findings[0]


def test_visit_nested_function_import_still_flagged():
    findings = _scan("def f():\n    import pwd\n")
    assert len(findings) == 1
    assert "pwd" in findings[0]
