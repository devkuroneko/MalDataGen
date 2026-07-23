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
            valid_x, valid_y = self._save_split(directory, "valid", numpy.ones((2, 4)), [1, 2])
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
            self.assertEqual(bundle.schema.num_features, 4)
            self.assertEqual(bundle.schema.classes, (0, 1, 2))
            self.assertEqual(bundle.schema.data_format, "npy_xy")
            self.assertEqual(bundle.schema.split_mode, "provided")
            self.assertEqual(bundle.schema.target_dtype, str(bundle.train.y.dtype))
            self.assertEqual(bundle.schema.train_feature_min, 0.0)
            self.assertEqual(bundle.schema.train_feature_max, 0.0)

    def test_raises_when_x_y_rows_are_incompatible(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 2)), [0, 1])

            with self.assertRaisesRegex(ValueError, "row mismatch"):
                NpyXYLoader(train_x, train_y).load()

    def test_raises_when_split_feature_widths_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((3, 2)), [0, 1, 2])
            valid_x, valid_y = self._save_split(directory, "valid", numpy.zeros((3, 3)), [0, 1, 2])

            with self.assertRaisesRegex(ValueError, "expected 2"):
                NpyXYLoader(train_x, train_y, valid_x_path=valid_x, valid_y_path=valid_y).load()

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

    def test_rejects_negative_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((2, 2)), [0, -1])

            with self.assertRaisesRegex(ValueError, "negative labels"):
                NpyXYLoader(train_x, train_y).load()

    def test_preserves_memmap_and_values_when_mmap_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            values = numpy.array([[0.25, -0.25], [0.5, -0.5]], dtype=numpy.float64)
            train_x, train_y = self._save_split(directory, "train", values, [0, 1])

            bundle = NpyXYLoader(train_x, train_y, mmap_mode="r").load()

            self.assertIsInstance(bundle.train.X, numpy.memmap)
            self.assertIsInstance(bundle.train.y, numpy.memmap)
            numpy.testing.assert_array_equal(bundle.train.X, values)

    def test_accepts_appclassnet_zero_to_199_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = numpy.arange(200, dtype=numpy.int16)
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((200, 3)), labels)

            bundle = NpyXYLoader(
                train_x,
                train_y,
                source_profile="appclassnet_top200",
                expected_num_features=3,
            ).load()

            self.assertEqual(bundle.schema.num_classes, 200)
            self.assertEqual(bundle.schema.classes[0], 0)
            self.assertEqual(bundle.schema.classes[-1], 199)

    def test_rejects_npy_y_with_more_than_one_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((2, 2)), [[0], [1]])

            with self.assertRaisesRegex(ValueError, "must be 1D"):
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
            self.assertEqual(bundle.metadata["label_mapping_original_to_zero_based"], {1: 0, 2: 1, 3: 2})

    def test_rejects_optional_split_without_y_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_split(directory, "train", numpy.zeros((2, 2)), [0, 1])
            valid_x = Path(directory) / "valid_x.npy"
            numpy.save(valid_x, numpy.zeros((1, 2)))

            with self.assertRaisesRegex(ValueError, "provided together"):
                NpyXYLoader(train_x, train_y, valid_x_path=valid_x)


if __name__ == "__main__":
    unittest.main()
