"""The public spec gate follows the spec standard: status.json is generated and never tracked."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_specs", ROOT / "scripts/check_public_specs.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


PAD = "This sentence pads the document past the minimum length the gate expects. " * 4
CONTENT = {
    "spec.md": "# Example\n\nThe acceptance requirement is stated here in a complete public document.\n\n"
    + PAD
    + "\n",
    "plan.md": "# Example plan\n\nThe design and architecture are described here.\n\n"
    + PAD
    + "\n",
    "test-spec.md": "# Example tests\n\nThe expected proof of each test is stated here.\n\n"
    + PAD
    + "\n",
    "tasks.md": "# Example tasks\n\n- [ ] T001 Do the work.\n\n" + PAD,
    "status.json": "{}\n",
}


class PublicSpecContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def spec(self, *names):
        paths = set()
        for name in names:
            path = Path("specs/example") / name
            (self.root / path.parent).mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(CONTENT[name])
            paths.add(path)
        return paths

    def test_a_spec_without_status_json_is_complete(self):
        paths = self.spec("spec.md", "plan.md", "test-spec.md", "tasks.md")
        self.assertEqual(checker._contract_errors(self.root, paths), [])

    def test_a_missing_authored_file_is_reported(self):
        paths = self.spec("spec.md", "plan.md", "tasks.md")
        errors = checker._contract_errors(self.root, paths)
        self.assertEqual(
            errors, ["specs/example/test-spec.md: missing required owner contract"]
        )

    def test_status_json_is_not_a_required_file(self):
        self.assertNotIn("status.json", checker.REQUIRED)

    def test_a_tracked_status_json_is_reported(self):
        paths = self.spec(
            "spec.md", "plan.md", "test-spec.md", "tasks.md", "status.json"
        )
        errors = checker._contract_errors(self.root, paths)
        self.assertEqual(
            errors,
            [
                "specs/example/status.json: status.json is generated; it must not be tracked"
            ],
        )


if __name__ == "__main__":
    unittest.main()
