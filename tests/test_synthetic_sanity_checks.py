import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.DataIO.SyntheticSanityChecks import NON_CLASS_CONDITIONAL_WARNING
from Engine.DataIO.SyntheticSanityChecks import SCALE_WARNING
from Engine.DataIO.SyntheticSanityChecks import SyntheticSanityChecker


class SyntheticSanityChecksTest(unittest.TestCase):

    def _checker(self, directory, real_x, real_y, synthetic_data, number_classes=2):
        checker = SyntheticSanityChecker(
            real_x=numpy.asarray(real_x, dtype=numpy.float32),
            real_y=numpy.asarray(real_y, dtype=numpy.int64),
            synthetic_data=synthetic_data,
            number_classes=number_classes,
            execution_mode="batches",
            arguments=SimpleNamespace(train_samples_per_class=10),
            fold_number=1,
            model_type="adversarial",
            experiment_directory=directory,
        )
        checker.output_path = Path(directory) / "synthetic_sanity_checks.json"
        return checker

    def test_writes_distribution_scale_distance_and_classifier_report(self):
        with tempfile.TemporaryDirectory() as directory:
            real_x = numpy.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=numpy.float32)
            real_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
            synthetic_data = {
                0: numpy.array([[0, 0], [0, 1]], dtype=numpy.float32),
                1: numpy.array([[10, 10], [10, 11]], dtype=numpy.float32),
            }

            output_path, report = self._checker(directory, real_x, real_y, synthetic_data).run()

            self.assertEqual(output_path, Path(directory) / "synthetic_sanity_checks.json")
            self.assertEqual(report["distribution_by_class"]["synthetic_counts_by_class"], {"0": 2, "1": 2})
            self.assertEqual(report["distribution_by_class"]["missing_classes"], [])
            self.assertEqual(report["scale"]["synthetic_outside_real_range_global_percent"], 0.0)
            self.assertEqual(report["real_to_synthetic_classifier"]["top1_accuracy"], 1.0)
            self.assertEqual(len(report["real_to_synthetic_classifier"]["confusion_matrix"]), 2)
            self.assertEqual(report["class_mean_distance"]["per_class"][0]["real_feature_mean"], [0.0, 0.5])
            self.assertEqual(report["class_mean_distance"]["per_class"][0]["synthetic_feature_mean"], [0.0, 0.5])
            self.assertEqual(report["class_mean_distance"]["worst_classes"][0]["l2_distance_between_means"], 0.0)

            with output_path.open() as sanity_file:
                persisted = json.load(sanity_file)
            self.assertEqual(persisted["status"], "completed")

    def test_warns_when_synthetic_values_are_outside_real_range(self):
        with tempfile.TemporaryDirectory() as directory:
            real_x = numpy.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=numpy.float32)
            real_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
            synthetic_data = {
                0: numpy.full((2, 2), 10, dtype=numpy.float32),
                1: numpy.full((2, 2), 10, dtype=numpy.float32),
            }

            _, report = self._checker(directory, real_x, real_y, synthetic_data).run()

            self.assertIn(SCALE_WARNING, report["warnings"])
            self.assertEqual(report["scale"]["synthetic_outside_real_range_global_percent"], 100.0)

    def test_warns_when_predictions_collapse_to_one_class(self):
        with tempfile.TemporaryDirectory() as directory:
            real_x = numpy.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=numpy.float32)
            real_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
            synthetic_data = {
                0: numpy.array([[0, 0], [0, 1]], dtype=numpy.float32),
                1: numpy.array([[0, 0], [0, 1]], dtype=numpy.float32),
            }

            _, report = self._checker(directory, real_x, real_y, synthetic_data).run()

            self.assertIn(NON_CLASS_CONDITIONAL_WARNING, report["warnings"])
            self.assertEqual(report["real_to_synthetic_classifier"]["dominant_predicted_class_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
