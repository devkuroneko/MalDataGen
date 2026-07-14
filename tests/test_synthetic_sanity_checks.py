import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.DataIO.SyntheticSanityChecks import NON_CLASS_CONDITIONAL_WARNING
from Engine.DataIO.SyntheticSanityChecks import SCALE_WARNING
from Engine.DataIO.SyntheticSanityChecks import SyntheticSanityChecker
from Engine.DataIO.SyntheticSanityChecks import _stratified_real_subset
from Engine.DataIO.DatasetContracts import AlignedDataset
from main import SynDataGen


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

    def test_real_x_smaller_than_real_y_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            real_x = numpy.zeros((2, 2), dtype=numpy.float32)
            real_y = numpy.array([0, 1, 1], dtype=numpy.int64)

            with self.assertRaisesRegex(ValueError, "X has 2 rows, y has 3 rows"):
                self._checker(directory, real_x, real_y, {0: real_x, 1: real_x}).run()

    def test_real_y_smaller_than_real_x_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            real_x = numpy.zeros((3, 2), dtype=numpy.float32)
            real_y = numpy.array([0, 1], dtype=numpy.int64)

            with self.assertRaisesRegex(ValueError, "X has 3 rows, y has 2 rows"):
                self._checker(directory, real_x, real_y, {0: real_x[:1], 1: real_x[:1]}).run()

    def test_fold_x_uses_fold_y(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = SynDataGen.__new__(SynDataGen)
            owner.fold_number = 0
            fold_x = numpy.zeros((4, 2), dtype=numpy.float32)
            fold_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)

            aligned = owner._aligned_sanity_real_dataset(fold_x, fold_y, "valid")

            self.assertEqual(aligned.X.shape[0], 4)
            self.assertEqual(aligned.y.tolist(), [0, 0, 1, 1])

    def test_subset_x_y_uses_same_indices(self):
        real_x = numpy.arange(20, dtype=numpy.float32).reshape(10, 2)
        real_y = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=numpy.int64)

        subset_x, subset_y, subset_counts = _stratified_real_subset(real_x, real_y, 2, 2)

        self.assertEqual(subset_x.shape[0], subset_y.shape[0])
        self.assertEqual(subset_counts.tolist(), [2, 2])

    def test_aligned_dataset_is_accepted_by_sanity_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            aligned = AlignedDataset(
                X=numpy.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=numpy.float32),
                y=numpy.array([0, 0, 1, 1], dtype=numpy.int64),
                split_name="valid",
                fold_id=1,
                source_indices=numpy.array([8, 9, 10, 11]),
            )
            synthetic_data = {
                0: numpy.array([[0, 0]], dtype=numpy.float32),
                1: numpy.array([[10, 10]], dtype=numpy.float32),
            }
            checker = SyntheticSanityChecker(
                real_x=aligned.X,
                real_y=aligned.y,
                synthetic_data=synthetic_data,
                number_classes=2,
                execution_mode="batches",
                arguments=SimpleNamespace(train_samples_per_class=1),
                fold_number=1,
                model_type="copy",
                experiment_directory=directory,
            )
            checker.output_path = Path(directory) / "synthetic_sanity_checks.json"

            _, report = checker.run()

            self.assertEqual(report["real_to_synthetic_classifier"]["real_subset_total_rows"], 2)

    def test_subset_by_classes_fails_before_masking_when_y_is_from_full_dataset(self):
        fold_x = numpy.zeros((4, 2), dtype=numpy.float32)
        full_y = numpy.array([0, 0, 1, 1, 0, 1], dtype=numpy.int64)

        with self.assertRaisesRegex(ValueError, "subset by classes input.*X has 4 rows, y has 6 rows"):
            SynDataGen._subset_by_classes(fold_x, full_y, [0, 1])

    def test_insufficient_synthetic_generation_plan_fails_fast(self):
        owner = SynDataGen.__new__(SynDataGen)
        owner.arguments = SimpleNamespace(
            synthetic_train_samples_per_class=500,
            synthetic_test_samples_per_class=500,
        )

        with self.assertRaisesRegex(ValueError, "InsufficientSyntheticGenerationPlan: class=0 required=1000 planned=256"):
            owner._validate_synthetic_generation_plan({
                "classes": {0: 256, 1: 1000},
                "number_classes": 2,
            })


if __name__ == "__main__":
    unittest.main()
