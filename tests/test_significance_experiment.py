import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.Evaluation.SignificanceExperiment import SignificanceConfig
from Engine.Evaluation.SignificanceExperiment import SignificanceExperimentRunner


class SignificanceExperimentTest(unittest.TestCase):

    def _data(self, samples_per_class, offset=0.0, noise=0.01):
        rng = numpy.random.default_rng(123 + samples_per_class + int(offset * 1000))
        x_parts = []
        y_parts = []
        for class_id in (0, 1):
            center = numpy.array([float(class_id) * 5.0, float(class_id), 1.0], dtype=numpy.float32)
            values = center + offset + rng.normal(scale=noise, size=(samples_per_class, 3)).astype(numpy.float32)
            x_parts.append(values)
            y_parts.append(numpy.full(samples_per_class, class_id, dtype=numpy.int64))
        return numpy.vstack(x_parts), numpy.concatenate(y_parts)

    def test_runner_writes_paired_significance_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._data(260, offset=0.0)
            test_x, test_y = self._data(560, offset=0.1)
            synthetic_x, synthetic_y = self._data(240, offset=0.02)

            config = SignificanceConfig(
                output_dir=directory,
                classifiers=["decision_tree_subset"],
                seeds=[0, 1, 2, 3, 4],
                num_classes=2,
                test_samples_per_class=500,
                command=["test-runner"],
            )
            paths = SignificanceExperimentRunner(
                train_x,
                train_y,
                test_x,
                test_y,
                synthetic_x,
                synthetic_y,
                config,
            ).run()

            for path in paths.values():
                self.assertTrue(Path(path).is_file())

            matrix = json.loads(Path(paths["experiment_matrix"]).read_text(encoding="utf-8"))
            self.assertEqual(matrix["seeds"], [0, 1, 2, 3, 4])
            self.assertEqual(matrix["status"], "completed")
            self.assertIn("git", matrix)
            self.assertIn("dataset_hashes", matrix)

            per_seed = _read_csv(paths["per_seed_results"])
            self.assertEqual(len(per_seed), 35)
            test_hashes_by_seed = {}
            for row in per_seed:
                test_hashes_by_seed.setdefault(row["seed"], set()).add(row["test_indices_hash"])
                self.assertIn("MacroF1", row)
                self.assertIn("confusion_matrix", row)
                self.assertEqual(int(row["effective_test_rows"]), 1000)
            self.assertTrue(all(len(values) == 1 for values in test_hashes_by_seed.values()))

            aggregate = _read_csv(paths["aggregate_results"])
            self.assertEqual(len(aggregate), 7)
            self.assertIn("MacroF1_ci95_low", aggregate[0])
            self.assertIn("memory_mb_delta_mean", aggregate[0])
            self.assertIn("effective_test_rows_mean", aggregate[0])

            paired = _read_csv(paths["paired_comparisons"])
            comparison_ids = {row["comparison_id"] for row in paired}
            self.assertIn("s200_vs_r200", comparison_ids)
            self.assertIn("label_permutation_vs_chance", comparison_ids)
            main_rows = [row for row in paired if row["comparison_family"] == "main_macro_f1"]
            self.assertTrue(all(row["p_value_holm"] != "" for row in main_rows))

            report = Path(paths["statistical_report"]).read_text(encoding="utf-8")
            self.assertIn("Paired Comparisons", report)

    def test_controls_are_equivalent_to_r200_within_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._data(260, offset=0.0)
            test_x, test_y = self._data(560, offset=0.1)
            synthetic_x, synthetic_y = self._data(240, offset=0.02)

            config = SignificanceConfig(
                output_dir=directory,
                classifiers=["decision_tree_subset"],
                seeds=[0, 1],
                num_classes=2,
                test_samples_per_class=500,
                command=["test-runner"],
            )
            paths = SignificanceExperimentRunner(
                train_x,
                train_y,
                test_x,
                test_y,
                synthetic_x,
                synthetic_y,
                config,
            ).run()

            by_seed_scenario = {
                (int(row["seed"]), row["scenario_id"]): row
                for row in _read_csv(paths["per_seed_results"])
            }
            for seed in (0, 1):
                r200 = by_seed_scenario[(seed, "r200_r500")]
                real_resample = by_seed_scenario[(seed, "real_resample_r500")]
                label_permutation = by_seed_scenario[(seed, "label_permutation_r500")]
                self.assertEqual(real_resample["train_hash"], r200["train_hash"])
                self.assertEqual(label_permutation["train_hash"], r200["train_hash"])
                self.assertNotEqual(label_permutation["train_class_counts"], "")

    def test_noninferiority_margin_requires_justification(self):
        train_x, train_y = self._data(260)
        test_x, test_y = self._data(560)
        synthetic_x, synthetic_y = self._data(240)
        config = SignificanceConfig(
            output_dir="/tmp/significance-test",
            classifiers=["decision_tree_subset"],
            seeds=[0, 1],
            num_classes=2,
            noninferiority_margin=0.02,
        )

        with self.assertRaisesRegex(ValueError, "justification"):
            SignificanceExperimentRunner(
                train_x,
                train_y,
                test_x,
                test_y,
                synthetic_x,
                synthetic_y,
                config,
            )


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
