import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.DataIO.NpyXYLoader import NpyXYLoader


class NpyXYLoaderTest(unittest.TestCase):

    def _save_split(self, directory, name, x_values, y_values):
        x_path = Path(directory) / f"{name}_x.npy"
        y_path = Path(directory) / f"{name}_y.npy"
        numpy.save(x_path, numpy.asarray(x_values))
        numpy.save(y_path, numpy.asarray(y_values))
        return x_path, y_path

    def test_loads_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(
                directory,
                "train",
                [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]],
                [0, 1, 1],
            )

            bundle = NpyXYLoader(train_x, train_y).load()

            self.assertEqual(bundle.train.name, "train")
            self.assertEqual(bundle.train.X.shape, (3, 2))
            self.assertEqual(bundle.train.y.shape, (3,))
            self.assertIsNone(bundle.valid)
            self.assertIsNone(bundle.test)
            self.assertEqual(bundle.schema.feature_names, ["f0", "f1"])
            self.assertEqual(bundle.schema.feature_type, "continuous")
            self.assertEqual(bundle.schema.target_type, "multiclass")
            self.assertEqual(bundle.schema.num_classes, 2)
            self.assertEqual(bundle.schema.source_format, "npy_xy")

    def test_loads_train_valid_test(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 4)), [0, 1, 2])
            valid_x, valid_y = self._save_split(directory, "valid", numpy.ones((2, 4)), [[1], [2]])
            test_x, test_y = self._save_split(directory, "test", numpy.full((1, 4), 2.0), [0])

            bundle = NpyXYLoader(
                train_x,
                train_y,
                valid_x_path=valid_x,
                valid_y_path=valid_y,
                test_x_path=test_x,
                test_y_path=test_y,
                dtype=numpy.float32,
            ).load()

            self.assertEqual(bundle.train.X.dtype, numpy.float32)
            self.assertEqual(bundle.valid.y.shape, (2,))
            self.assertEqual(bundle.test.X.shape, (1, 4))
            self.assertEqual(set(bundle.splits.keys()), {"train", "valid", "test"})
            self.assertEqual(bundle.schema.num_classes, 3)

    def test_raises_when_x_y_rows_are_incompatible(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 2)), [0, 1])

            with self.assertRaisesRegex(ValueError, "row mismatch"):
                NpyXYLoader(train_x, train_y).load()

    def test_infers_twenty_feature_names(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((5, 20)), [0, 1, 2, 3, 4])

            bundle = NpyXYLoader(train_x, train_y).load()

            self.assertEqual(bundle.train.num_features, 20)
            self.assertEqual(bundle.schema.feature_names[0], "f0")
            self.assertEqual(bundle.schema.feature_names[-1], "f19")
            self.assertEqual(len(bundle.schema.feature_names), 20)

    def test_infers_two_hundred_classes_from_sparse_zero_based_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 20)), [0, 50, 199])

            bundle = NpyXYLoader(train_x, train_y).load()

            self.assertEqual(bundle.schema.num_classes, 200)
            self.assertIsNone(bundle.schema.class_labels)

    def test_rejects_non_integer_multiclass_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((2, 2)), [0.0, 1.5])

            with self.assertRaisesRegex(ValueError, "integer labels"):
                NpyXYLoader(train_x, train_y).load()

    def test_warns_for_one_based_labels_without_remapping(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 2)), [1, 2, 3])

            with self.assertLogs(level="WARNING") as logs:
                bundle = NpyXYLoader(train_x, train_y).load()

            self.assertIn("1-based", "\n".join(logs.output))
            self.assertEqual(bundle.train.y.tolist(), [1, 2, 3])
            self.assertEqual(bundle.schema.num_classes, 3)
            self.assertEqual(bundle.schema.class_labels, (1, 2, 3))

    def test_can_remap_one_based_labels_to_zero_based_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 2)), [1, 2, 3])

            with self.assertLogs(level="WARNING") as logs:
                bundle = NpyXYLoader(train_x, train_y, remap_labels_to_zero_based=True).load()

            self.assertIn("Remapping labels", "\n".join(logs.output))
            self.assertEqual(bundle.train.y.tolist(), [0, 1, 2])
            self.assertEqual(bundle.schema.num_classes, 3)
            self.assertIsNone(bundle.schema.class_labels)

    def test_rejects_optional_split_without_y_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((2, 2)), [0, 1])
            valid_x = Path(directory) / "valid_x.npy"
            numpy.save(valid_x, numpy.zeros((1, 2)))

            with self.assertRaisesRegex(ValueError, "provided together"):
                NpyXYLoader(train_x, train_y, valid_x_path=valid_x)


if __name__ == "__main__":
    unittest.main()
