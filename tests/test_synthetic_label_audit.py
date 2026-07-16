import json
import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.DataIO.SyntheticLabelAudit import SyntheticLabelGenerationAudit


class SyntheticLabelAuditTest(unittest.TestCase):

    def test_audit_writes_records_and_class_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = SyntheticLabelGenerationAudit(
                execution_mode="batches",
                number_classes=2,
                label_mapping={0: 0, 1: 1},
                fold_number=1,
                model_type="adversarial",
            )
            audit.output_path = Path(directory) / "label_generation_audit.json"

            audit.record(0, 0, numpy.array([[0.0, 1.0], [2.0, 3.0]], dtype=numpy.float32), batch_index=0)
            audit.record(1, 1, numpy.array([[4.0, 5.0]], dtype=numpy.float32), batch_index=0)

            output_path, report = audit.finalize()

            self.assertEqual(output_path, audit.output_path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["records"][0]["requested_class"], 0)
            self.assertEqual(report["records"][0]["saved_label"], 0)
            self.assertEqual(report["records"][0]["generated_count"], 2)
            self.assertEqual(report["class_summary"]["0"]["generated_count"], 2)
            self.assertEqual(report["class_summary"]["1"]["feature_mean"], [4.0, 5.0])

            with audit.output_path.open() as audit_file:
                persisted = json.load(audit_file)
            self.assertEqual(persisted["status"], "passed")

    def test_audit_fails_on_requested_saved_label_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = SyntheticLabelGenerationAudit(
                execution_mode="normal",
                number_classes=2,
            )
            audit.output_path = Path(directory) / "label_generation_audit.json"

            audit.record(0, 1, numpy.ones((1, 2), dtype=numpy.float32))

            with self.assertRaises(ValueError):
                audit.finalize()

            with audit.output_path.open() as audit_file:
                persisted = json.load(audit_file)
            self.assertEqual(persisted["status"], "failed")
            self.assertTrue(persisted["errors"])

    def test_expected_classes_enforce_requested_subset_only(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = SyntheticLabelGenerationAudit(
                execution_mode="batches",
                number_classes=200,
                expected_classes=[0, 1],
            )
            audit.output_path = Path(directory) / "label_generation_audit.json"
            audit.record(0, 0, numpy.ones((1, 2), dtype=numpy.float32))

            with self.assertRaises(ValueError):
                audit.finalize()

            with audit.output_path.open() as audit_file:
                persisted = json.load(audit_file)
            self.assertIn("missing", persisted["errors"][0])

    def test_appclassnet_200_allows_partial_subset_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = SyntheticLabelGenerationAudit(
                execution_mode="batches",
                number_classes=200,
                expected_classes=[0],
            )
            audit.output_path = Path(directory) / "label_generation_audit.json"
            audit.record(0, 0, numpy.ones((1, 2), dtype=numpy.float32))

            _, report = audit.finalize()

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["expected_classes"], [0])


if __name__ == "__main__":
    unittest.main()
