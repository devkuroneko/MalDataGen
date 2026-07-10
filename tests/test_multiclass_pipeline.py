import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.DataIO.LabelUtils import build_class_metadata
from Engine.DataIO.LabelUtils import one_hot_encode_labels
from Engine.DataIO.NpyXYLoader import NpyXYLoader
from Engine.Evaluation.CrossValidation import _apply_bundle_to_owner
from Engine.Evaluation.CrossValidation import load_dataset_from_args
from Engine.Metrics.Metrics import Metrics


class MulticlassPipelineTest(unittest.TestCase):

    def _save_train(self, directory, labels, num_features=4):
        x_path = Path(directory) / "train_x.npy"
        y_path = Path(directory) / "train_y.npy"
        numpy.save(x_path, numpy.zeros((len(labels), num_features), dtype=numpy.float32))
        numpy.save(y_path, numpy.asarray(labels))
        return str(x_path), str(y_path)

    def _args(self, directory, num_classes=None, remap=False):
        return SimpleNamespace(
            data_format="npy_xy",
            train_x_path=str(Path(directory) / "train_x.npy"),
            train_y_path=str(Path(directory) / "train_y.npy"),
            valid_x_path=None,
            valid_y_path=None,
            test_x_path=None,
            test_y_path=None,
            mmap_npy=True,
            target_type="multiclass",
            feature_type="continuous",
            num_classes=num_classes,
            remap_labels_to_zero_based=remap,
            split_mode="cross_validation",
            data_type="continuous",
            number_samples_per_class={"classes": {0: 1}, "number_classes": 1},
            number_k_folds=1,
        )

    def test_three_classes_infer_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_train(directory, [0, 1, 2])

            bundle = load_dataset_from_args(self._args(directory))

            self.assertEqual(bundle.schema.num_classes, 3)
            self.assertEqual(bundle.train.y.tolist(), [0, 1, 2])

    def test_ten_classes_infer_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_train(directory, list(range(10)))

            bundle = load_dataset_from_args(self._args(directory))

            self.assertEqual(bundle.schema.num_classes, 10)
            self.assertIsNone(bundle.schema.class_labels)

    def test_top_200_sparse_labels_respect_user_num_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_train(directory, [0, 3, 199], num_features=20)

            args = self._args(directory, num_classes=200)
            bundle = load_dataset_from_args(args)
            owner = SimpleNamespace(arguments=args, list_folds=[])
            _apply_bundle_to_owner(owner, bundle)

            self.assertEqual(bundle.schema.num_classes, 200)
            self.assertEqual(owner._number_samples_per_class["number_classes"], 200)
            self.assertEqual(owner._data_loaded_labels.ndim, 1)

    def test_one_based_labels_warn_without_remap_and_one_hot_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_train(directory, [1, 2, 3])

            with self.assertLogs(level="WARNING") as logs:
                bundle = NpyXYLoader(train_x, train_y).load()

            self.assertIn("1-based", "\n".join(logs.output))
            self.assertEqual(bundle.train.y.tolist(), [1, 2, 3])
            with self.assertRaisesRegex(ValueError, "num_classes=3"):
                one_hot_encode_labels(bundle.train.y, num_classes=bundle.schema.num_classes)

    def test_one_based_labels_remap_when_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            train_x, train_y = self._save_train(directory, [1, 2, 3])

            bundle = NpyXYLoader(train_x, train_y, remap_labels_to_zero_based=True).load()

            self.assertEqual(bundle.train.y.tolist(), [0, 1, 2])
            self.assertEqual(bundle.schema.num_classes, 3)

    def test_class_metadata_uses_domain_not_observed_count(self):
        metadata = build_class_metadata([0, 199], num_classes=200, data_type="continuous")

        self.assertEqual(metadata["classes"], {0: 1, 199: 1})
        self.assertEqual(metadata["number_classes"], 200)
        self.assertEqual(metadata["data_type"], "continuous")

    def test_multiclass_metrics_are_selected_by_target_type(self):
        metrics = Metrics.__new__(Metrics)
        metrics._target_type = "multiclass"
        metrics._data_type = "binary"
        metrics._dictionary_binary_metrics = {"LegacyBinaryMetric": object()}

        self.assertEqual(
            metrics._get_classifier_metric_names(),
            [
                "Accuracy",
                "MacroPrecision",
                "MacroRecall",
                "MacroF1",
                "WeightedPrecision",
                "WeightedRecall",
                "WeightedF1",
                "BalancedAccuracy",
            ],
        )

        values = metrics._get_multiclass_metric_values([0, 1, 2], [0, 2, 2])
        self.assertIn("MacroF1", values)
        self.assertIn("WeightedF1", values)


if __name__ == "__main__":
    unittest.main()
