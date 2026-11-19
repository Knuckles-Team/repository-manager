"""The public spec gate accepts generated v2 without relaxing legacy receipts."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_specs", ROOT / "scripts/check_public_specs.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class PublicSpecSchemaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = Path("specs/example/status.json")
        (self.root / self.path.parent).mkdir(parents=True)
        (self.root / self.path.parent / "requirements.md").write_text("`RM-EXAMPLE-01`")
        self.document: dict[str, Any] = {
            "schema_version": 2,
            "spec_id": "RM-EXAMPLE",
            "owner_repo": "repository-manager",
            "delivery_state": "LANDED",
            "requirement_ids": ["RM-EXAMPLE-01"],
            "requirements": [
                {
                    "id": "RM-EXAMPLE-01",
                    "title": "Keep provenance",
                    "delivery_state": "LANDED",
                    "landed_in": ["123456789abc"],
                    "verified_by": [],
                }
            ],
        }

    def errors(self, document):
        (self.root / self.path).write_text(json.dumps(document))
        return checker._status_errors(self.root, self.path)

    def test_generated_v2_preserves_ids_states_and_provenance(self):
        before = copy.deepcopy(self.document)
        self.assertEqual(self.errors(self.document), [])
        self.assertEqual(self.document, before)
        self.assertEqual(json.loads((self.root / self.path).read_text()), before)

    def test_invalid_generated_structure_is_rejected(self):
        for field, value in [
            ("owner_repo", "another-repo"),
            ("delivery_state", "CLOSED"),
            ("requirement_ids", ["RM-EXAMPLE-01", "RM-EXAMPLE-01"]),
            ("requirements", ["invalid"]),
        ]:
            with self.subTest(field=field):
                document = copy.deepcopy(self.document)
                document[field] = value
                self.assertTrue(self.errors(document))

    def test_malformed_generated_provenance_is_rejected(self):
        for field in ("landed_in", "verified_by"):
            with self.subTest(field=field):
                document = copy.deepcopy(self.document)
                document["requirements"][0][field] = "not-an-array"
                self.assertTrue(self.errors(document))

    def test_legacy_evidence_rules_remain_enforced(self):
        document = copy.deepcopy(self.document)
        document["schema_version"] = 1
        errors = self.errors(document)
        self.assertTrue(any("evidence" in error for error in errors))
        self.assertTrue(any("acceptance_state" in error for error in errors))

    def test_actual_generated_specs_keep_their_content(self):
        for path in sorted((ROOT / "specs").glob("*/status.json")):
            if path.parent.name == "_template":
                continue
            with self.subTest(spec=path.parent.name):
                before = path.read_bytes()
                self.assertEqual(
                    checker._status_errors(ROOT, path.relative_to(ROOT)), []
                )
                self.assertEqual(path.read_bytes(), before)


class ComputedStatusTests(unittest.TestCase):
    """status.json is computed at read time and never tracked, so it is not required."""

    BODIES = {
        "spec.md": "# Example\n\nRequirement FR-001: the acceptance behaviour is described here.\n",
        "plan.md": "# Example plan\n\nThe architecture and design of the interface are described here.\n",
        "test-spec.md": "# Example tests\n\nEach test names the expected proof for its requirement.\n",
        "tasks.md": "# Example tasks\n\n- [ ] T001 implement the requirement and its bound test.\n",
    }

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        feature = self.root / "specs" / "example"
        feature.mkdir(parents=True)
        for name, body in self.BODIES.items():
            (feature / name).write_text(body + "Substantive detail. " * 12)

    def paths(self):
        return set(checker.tracked_specs(self.root))

    def test_a_feature_without_status_json_passes(self):
        self.assertEqual(checker._contract_errors(self.root, self.paths()), [])

    def test_a_missing_owner_document_is_still_a_finding(self):
        (self.root / "specs/example/plan.md").unlink()
        errors = checker._contract_errors(self.root, self.paths())
        self.assertEqual(
            errors, ["specs/example/plan.md: missing required owner contract"]
        )

    def test_a_present_generated_status_is_still_validated(self):
        (self.root / "specs/example/status.json").write_text("[]")
        errors = checker._contract_errors(self.root, self.paths())
        self.assertEqual(
            errors, ["specs/example/status.json: status must be an object"]
        )

    def test_the_repository_specs_pass_without_a_tracked_status(self):
        self.assertEqual(checker.problems(ROOT), [])
