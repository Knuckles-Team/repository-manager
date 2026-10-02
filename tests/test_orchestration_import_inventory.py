"""Standalone stdlib tests; also collected by the repository's pytest suite.

Run directly with python tests/test_orchestration_import_inventory.py to avoid
initializing the unrelated native graph session in the shared pytest conftest.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = runpy.run_path(str(ROOT / "scripts/check_orchestration_import_inventory.py"))


def observations(source: str) -> dict:
    imports, unresolved = CHECK["source_observations"]("sample.py", source.encode())
    return {
        "schema_version": 1,
        "namespaces": list(CHECK["NAMESPACES"]),
        "imports": imports,
        "unresolved": unresolved,
    }


def reviewed(actual: dict, category: str = "orchestration") -> dict:
    result = copy.deepcopy(actual)
    for row in result["imports"]:
        row.update(
            classification=category,
            reviewer="fixture reviewer",
            evidence="Fixture import exercises a reviewed boundary.",
        )
    return result


class ImportInventoryTests(unittest.TestCase):
    def test_repeated_imports_on_one_line_are_distinct_occurrences(self) -> None:
        actual = observations("import agent_utilities as a, agent_utilities as b")
        self.assertEqual(len(actual["imports"]), 2)
        self.assertEqual(len({CHECK["identity"](r) for r in actual["imports"]}), 2)
        self.assertEqual(CHECK["validate"](actual, reviewed(actual)), [])

    def test_duplicate_json_keys_cannot_replace_a_classification(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            json.loads(
                '{"classification":"governance","classification":"obsolete"}',
                object_pairs_hook=CHECK["unique_fields"],
            )

    def test_static_import_occurrences_and_both_namespaces(self) -> None:
        actual = observations(
            "import agent_utilities as au, agent_orchestration.transport\n"
            "from agent_utilities.governance import lanes as lane, queue\n"
            "from agent_orchestration import *\n"
            "import agent_utilities_extra\nfrom .agent_utilities import local\n"
        )
        self.assertEqual(len(actual["imports"]), 5)
        self.assertEqual(len({CHECK["identity"](r) for r in actual["imports"]}), 5)
        self.assertEqual(CHECK["validate"](actual, reviewed(actual)), [])

    def test_literal_dynamic_syntax_and_alias_chains(self) -> None:
        sources = [
            'import importlib\nimportlib.import_module("agent_utilities.x")',
            'import importlib as il\nil.import_module(name="agent_utilities.x")',
            'from importlib import import_module as load\nload("agent_utilities.x")',
            '__import__(name="agent_utilities.x")',
            'import builtins as b\nb.__import__("agent_utilities.x")',
            'from builtins import __import__ as load\nload("agent_utilities.x")',
            'import importlib\nload = importlib.import_module\nagain = load\nagain("agent_utilities.x")',
            'load = __import__\nload("agent_utilities.x")',
            'import importlib\nil = importlib\nil.import_module("agent_utilities.x")',
            'import importlib\nload: object = importlib.import_module\nload("agent_utilities.x")',
            'def f():\n import importlib as il\n il.import_module("agent_utilities.x")',
            'import importlib\nimportlib.import_module(".x", package="agent_utilities")',
        ]
        for source in sources:
            with self.subTest(source=source):
                actual = observations(source)
                self.assertEqual(
                    [r["target"] for r in actual["imports"]], ["agent_utilities.x"]
                )
                self.assertEqual(actual["unresolved"], [])

    def test_nonliteral_missing_and_relative_targets_fail_visibly(self) -> None:
        calls = [
            "load(variable)",
            "load()",
            "load(name=variable)",
            'load("agent_" + variable)',
            'load(f"agent_utilities.{name}")',
            'load(".x", package=variable)',
            'load("...x", "agent_utilities")',
            "load(*args)",
            "load(**kwargs)",
        ]
        for call in calls:
            with self.subTest(call=call):
                actual = observations(
                    "from importlib import import_module as load\n" + call
                )
                self.assertEqual(len(actual["unresolved"]), 1)
                self.assertIn(
                    "unresolved dynamic target",
                    CHECK["validate"](actual, reviewed(actual))[0],
                )
        actual = observations('__import__("x", level=1)')
        self.assertEqual(len(actual["unresolved"]), 1)

    def test_unrelated_calls_and_strings_are_not_imports(self) -> None:
        actual = observations(
            'def import_module(name): return name\nimport_module("agent_utilities.x")\n'
            'text = "import agent_utilities"\n# import agent_utilities\n'
            'import importlib\nimportlib.import_module("json")\n'
        )
        self.assertEqual(actual["imports"], [])
        self.assertEqual(actual["unresolved"], [])

    def test_alias_rebinding_cannot_hide_unresolved_call(self) -> None:
        actual = observations(
            "import importlib\nload = importlib.import_module\nload = replacement\nload(target)"
        )
        self.assertEqual(len(actual["unresolved"]), 1)

    def test_each_category_and_invalid_or_missing_review(self) -> None:
        actual = observations("import agent_utilities")
        for category in CHECK["CATEGORIES"]:
            self.assertEqual(CHECK["validate"](actual, reviewed(actual, category)), [])
        for field, value in [
            ("classification", None),
            ("classification", ["governance"]),
            ("classification", "unknown"),
            ("reviewer", ""),
            ("evidence", " "),
        ]:
            with self.subTest(field=field, value=value):
                inventory = reviewed(actual)
                inventory["imports"][0][field] = value
                self.assertTrue(CHECK["validate"](actual, inventory))

    def test_missing_duplicate_and_stale_dispositions(self) -> None:
        actual = observations("import agent_utilities")
        inventory = reviewed(actual)
        inventory["imports"] = []
        self.assertIn("missing classification", CHECK["validate"](actual, inventory)[0])
        inventory = reviewed(actual)
        inventory["imports"] *= 2
        self.assertIn(
            "duplicate classification", CHECK["validate"](actual, inventory)[0]
        )
        changed = observations("import agent_utilities\n# changed context\n")
        self.assertIn(
            "stale classification", CHECK["validate"](changed, reviewed(actual))[0]
        )
        removed = observations("import json")
        self.assertIn(
            "stale classification", CHECK["validate"](removed, reviewed(actual))[0]
        )

    def test_discovery_covers_untracked_tests_and_never_executes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "tests").mkdir()
            source = root / "tests/example.py"
            source.write_text(
                'raise RuntimeError("must not execute")\nimport agent_utilities\n'
            )
            first = CHECK["discover"](root)
            self.assertEqual(first, CHECK["discover"](root))
            self.assertEqual(len(first["imports"]), 1)
            source.write_text("invalid python !!!")
            with self.assertRaises(SyntaxError):
                CHECK["discover"](root)

    def test_symlink_and_deleted_tracked_source_refuse_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            target = root / "source.py"
            target.write_text("import agent_utilities")
            link = root / "link.py"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "regular"):
                CHECK["discover"](root)
            link.unlink()
            subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
            target.unlink()
            with self.assertRaises(FileNotFoundError):
                CHECK["discover"](root)

    def test_cli_success_missing_inventory_and_unresolved_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "source.py"
            source.write_text("import agent_utilities")
            inventory = root / "inventory.json"
            args = ["--root", str(root), "--inventory", str(inventory)]
            with (
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(CHECK["main"](args), 2)
                inventory.write_text(json.dumps(reviewed(CHECK["discover"](root))))
                self.assertEqual(CHECK["main"](args), 0)
                source.write_text("__import__(name)")
                inventory.write_text(json.dumps(reviewed(CHECK["discover"](root))))
                self.assertEqual(CHECK["main"](args), 1)
                self.assertEqual(CHECK["main"](["--root", str(root), "--discover"]), 1)
                inventory.write_text("{}")
                self.assertEqual(CHECK["main"](args), 2)


if __name__ == "__main__":
    unittest.main()
