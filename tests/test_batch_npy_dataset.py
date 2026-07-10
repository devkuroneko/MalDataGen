import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.DataIO.BatchNpyDataset import BatchNpyDataset


class BatchNpyDatasetTest(unittest.TestCase):

    def _save_xy(self, directory, x_values, y_values):
        x_path = Path(directory) / "x.npy"
        y_path = Path(directory) / "y.npy"
        numpy.save(x_path, numpy.asarray(x_values))
        numpy.save(y_path, numpy.asarray(y_values))
        return x_path, y_path

    def test_iterates_batches_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(
                directory,
                numpy.arange(24, dtype=numpy.float32).reshape(6, 4),
                numpy.arange(6),
            )

            dataset = BatchNpyDataset(x_path, y_path, batch_size=2, mmap_mode="r")
            batches = list(dataset.iter_batches())

            self.assertEqual(dataset.num_samples, 6)
            self.assertEqual(dataset.num_features, 4)
            self.assertEqual(dataset.shape, (6, 4))
            self.assertEqual(len(batches), 3)
            numpy.testing.assert_array_equal(batches[0][0], numpy.arange(8, dtype=numpy.float32).reshape(2, 4))
            numpy.testing.assert_array_equal(batches[0][1], [0, 1])

    def test_last_batch_can_be_smaller(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(directory, numpy.zeros((5, 3)), numpy.arange(5))

            dataset = BatchNpyDataset(x_path, y_path, batch_size=2, mmap_mode="r")
            batches = list(dataset.iter_batches())

            self.assertEqual([batch_x.shape[0] for batch_x, _ in batches], [2, 2, 1])
            numpy.testing.assert_array_equal(batches[-1][1], [4])

    def test_max_samples_limits_iteration_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(directory, numpy.arange(30).reshape(10, 3), numpy.arange(10))

            dataset = BatchNpyDataset(x_path, y_path, batch_size=4, mmap_mode="r", max_samples=6)
            batches = list(dataset.iter_batches())

            self.assertEqual(dataset.num_samples, 6)
            self.assertEqual([batch_x.shape[0] for batch_x, _ in batches], [4, 2])
            numpy.testing.assert_array_equal(numpy.concatenate([batch_y for _, batch_y in batches]), numpy.arange(6))

    def test_max_samples_per_class_limits_each_class(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
            x_path, y_path = self._save_xy(directory, numpy.arange(18).reshape(9, 2), labels)

            with self.assertLogs(level="WARNING") as logs:
                dataset = BatchNpyDataset(x_path, y_path, batch_size=10, mmap_mode="r", max_samples_per_class=2)

            self.assertIn("mmap y scan", "\n".join(logs.output))
            self.assertEqual(dataset.num_samples, 6)
            self.assertEqual(dataset.num_classes, 3)
            self.assertEqual(dataset.selection_report["selected_counts_by_class"], {"0": 2, "1": 2, "2": 2})
            _, batch_y = next(dataset.iter_batches())
            unique, counts = numpy.unique(batch_y, return_counts=True)
            self.assertEqual(dict(zip(unique.tolist(), counts.tolist())), {0: 2, 1: 2, 2: 2})

    def test_mmap_mode_keeps_arrays_memory_mapped(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(directory, numpy.zeros((3, 2)), numpy.array([[0], [1], [2]]))

            dataset = BatchNpyDataset(x_path, y_path, batch_size=2, mmap_mode="r")

            self.assertIsInstance(dataset.X, numpy.memmap)
            self.assertIsInstance(dataset.y, numpy.memmap)
            self.assertEqual(dataset.y.shape, (3,))

    def test_head_returns_debug_rows_with_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(directory, numpy.arange(12, dtype=numpy.float64).reshape(4, 3), [0, 1, 2, 3])

            dataset = BatchNpyDataset(x_path, y_path, batch_size=2, mmap_mode="r", dtype=numpy.float32)
            head_x, head_y = dataset.head(2)

            self.assertEqual(head_x.dtype, numpy.float32)
            numpy.testing.assert_array_equal(head_x, numpy.array([[0, 1, 2], [3, 4, 5]], dtype=numpy.float32))
            numpy.testing.assert_array_equal(head_y, [0, 1])

    def test_shuffle_does_not_change_batch_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(directory, numpy.arange(40).reshape(20, 2), numpy.arange(20))

            dataset = BatchNpyDataset(x_path, y_path, batch_size=7, mmap_mode="r", shuffle=True, seed=7)
            labels = numpy.concatenate([batch_y for _, batch_y in dataset.iter_batches()])

            self.assertEqual(labels.shape, (20,))
            numpy.testing.assert_array_equal(numpy.sort(labels), numpy.arange(20))
            self.assertFalse(numpy.array_equal(labels, numpy.arange(20)))

    def test_simulated_two_hundred_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = numpy.arange(200)
            x_path, y_path = self._save_xy(directory, numpy.zeros((200, 20), dtype=numpy.float32), labels)

            dataset = BatchNpyDataset(x_path, y_path, batch_size=64, mmap_mode="r")

            self.assertEqual(dataset.num_classes, 200)
            self.assertEqual(dataset.classes_[0], 0)
            self.assertEqual(dataset.classes_[-1], 199)
            self.assertEqual(sum(batch_x.shape[0] for batch_x, _ in dataset.iter_batches()), 200)

    def test_rejects_incompatible_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            x_path, y_path = self._save_xy(directory, numpy.zeros((3, 2)), [0, 1])

            with self.assertRaisesRegex(ValueError, "row mismatch"):
                BatchNpyDataset(x_path, y_path, batch_size=2)


if __name__ == "__main__":
    unittest.main()
