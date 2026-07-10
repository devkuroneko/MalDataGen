import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.DataIO.StratifiedNpySelection import build_minimum_coverage_report
from Engine.DataIO.StratifiedNpySelection import get_last_stratified_selection_report
from Engine.DataIO.StratifiedNpySelection import select_stratified_indices_from_npy


class StratifiedNpySelectionTest(unittest.TestCase):

    def _save_y(self, directory, labels):
        y_path = Path(directory) / "y.npy"
        numpy.save(y_path, numpy.asarray(labels, dtype=numpy.int64))
        return y_path

    def test_selects_reproducible_stratified_indices_after_scanning_full_file(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = numpy.array([0] * 20 + [1] * 20 + [2], dtype=numpy.int64)
            y_path = self._save_y(directory, labels)

            first = select_stratified_indices_from_npy(y_path, samples_per_class=3, num_classes=3, seed=7)
            first_report = get_last_stratified_selection_report()
            second = select_stratified_indices_from_npy(y_path, samples_per_class=3, num_classes=3, seed=7)

            numpy.testing.assert_array_equal(first, second)
            selected_labels = labels[first]
            unique, counts = numpy.unique(selected_labels, return_counts=True)
            self.assertEqual(dict(zip(unique.tolist(), counts.tolist())), {0: 3, 1: 3, 2: 1})
            self.assertIn(40, first.tolist())
            self.assertEqual(first_report["total_rows_scanned"], 41)
            self.assertEqual(first_report["classes_below_limit"], [2])
            self.assertEqual(first_report["classes_absent"], [])

    def test_reports_absent_and_below_minimum_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            y_path = self._save_y(directory, [0, 0, 1])

            indices = select_stratified_indices_from_npy(y_path, samples_per_class=2, num_classes=4, seed=0)
            report = get_last_stratified_selection_report()
            coverage = build_minimum_coverage_report(report, min_samples_per_class_required=1)

            self.assertEqual(indices.shape[0], 3)
            self.assertEqual(report["selected_counts_by_class"], {"0": 2, "1": 1, "2": 0, "3": 0})
            self.assertEqual(report["classes_absent"], [2, 3])
            self.assertFalse(coverage["passes_minimum"])
            self.assertEqual(coverage["classes_below_minimum"], [2, 3])


if __name__ == "__main__":
    unittest.main()
