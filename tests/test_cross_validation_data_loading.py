import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.Evaluation.CrossValidation import _apply_bundle_to_owner
from Engine.Evaluation.CrossValidation import _build_provided_split_folds
from Engine.Evaluation.CrossValidation import StratifiedData
from Engine.Evaluation.CrossValidation import load_dataset_from_args


class DummyCsvOwner:
    def __init__(self):
        self.load_csv_called = False

    def load_csv(self):
        self.load_csv_called = True


def csv_args():
    return SimpleNamespace(
        data_format="csv",
        data_load_label_column=-1,
        data_load_max_samples=-1,
        data_load_max_columns=-1,
        data_load_start_column=0,
        data_load_end_column=-1,
        data_load_path_file_input="dataset.csv",
        data_load_path_file_output="OutputDir",
        data_load_exclude_columns=-1,
        data_type="binary",
        number_samples_per_class={"classes": {0: 1}, "number_classes": 1},
    )


def npy_args(directory, split_mode="cross_validation", with_valid=False, with_test=False):
    args = SimpleNamespace(
        data_format="npy_xy",
        train_x_path=str(Path(directory) / "train_x.npy"),
        train_y_path=str(Path(directory) / "train_y.npy"),
        valid_x_path=str(Path(directory) / "valid_x.npy") if with_valid else None,
        valid_y_path=str(Path(directory) / "valid_y.npy") if with_valid else None,
        test_x_path=str(Path(directory) / "test_x.npy") if with_test else None,
        test_y_path=str(Path(directory) / "test_y.npy") if with_test else None,
        mmap_npy=True,
        target_type="multiclass",
        feature_type="continuous",
        num_classes=None,
        remap_labels_to_zero_based=False,
        split_mode=split_mode,
        data_type="continuous",
        number_samples_per_class={"classes": {0: 1}, "number_classes": 1},
        number_k_folds=1,
    )
    return args


class CrossValidationDataLoadingTest(unittest.TestCase):

    def _save_split(self, directory, name, x_values, y_values):
        numpy.save(Path(directory) / f"{name}_x.npy", numpy.asarray(x_values))
        numpy.save(Path(directory) / f"{name}_y.npy", numpy.asarray(y_values))

    def test_csv_format_uses_legacy_owner_load(self):
        owner = DummyCsvOwner()
        result = load_dataset_from_args(csv_args(), owner=owner)

        self.assertIsNone(result)
        self.assertTrue(owner.load_csv_called)
        self.assertEqual(owner._data_load_path_file_input, "dataset.csv")

    def test_npy_xy_format_returns_dataset_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(directory, "train", numpy.zeros((3, 20)), [0, 1, 199])

            bundle = load_dataset_from_args(npy_args(directory))

            self.assertEqual(bundle.train.X.shape, (3, 20))
            self.assertEqual(bundle.train.y.tolist(), [0, 1, 199])
            self.assertEqual(bundle.schema.num_classes, 200)
            self.assertEqual(bundle.schema.feature_type, "continuous")
            self.assertEqual(bundle.metadata["mmap_mode"], "r")

    def test_provided_split_uses_valid_as_evaluation_without_kfold(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(directory, "train", numpy.zeros((3, 2)), [0, 1, 2])
            self._save_split(directory, "valid", numpy.ones((2, 2)), [1, 2])

            args = npy_args(directory, split_mode="provided", with_valid=True)
            bundle = load_dataset_from_args(args)
            owner = SimpleNamespace(arguments=args, list_folds=[])

            _apply_bundle_to_owner(owner, bundle)
            _build_provided_split_folds(owner, bundle)

            self.assertEqual(len(owner.list_folds), 1)
            fold = owner.list_folds[0]
            self.assertEqual(fold["x_training_real"].shape, (3, 2))
            self.assertEqual(fold["x_evaluation_real"].shape, (2, 2))
            self.assertEqual(fold["evaluation_split_name"], "valid")
            self.assertFalse(fold["evaluation_not_applicable"])

    def test_provided_train_only_marks_evaluation_not_applicable(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(directory, "train", numpy.zeros((3, 2)), [0, 1, 2])

            args = npy_args(directory, split_mode="provided")
            bundle = load_dataset_from_args(args)
            owner = SimpleNamespace(arguments=args, list_folds=[])

            _apply_bundle_to_owner(owner, bundle)
            with self.assertLogs(level="WARNING") as logs:
                _build_provided_split_folds(owner, bundle)

            self.assertIn("not_applicable", "\n".join(logs.output))
            self.assertEqual(len(owner.list_folds), 1)
            self.assertTrue(owner.list_folds[0]["evaluation_not_applicable"])
            self.assertEqual(owner.list_folds[0]["x_evaluation_real"].shape, (3, 2))

    def test_npy_cross_validation_preserves_schema_num_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(
                directory,
                "train",
                numpy.zeros((6, 20)),
                [0, 0, 1, 1, 199, 199],
            )

            args = npy_args(directory)
            args.number_k_folds = 2
            owner = SimpleNamespace(arguments=args, directory_output_data=directory)

            wrapped = StratifiedData(lambda self: "done")
            result = wrapped(owner)

            self.assertEqual(result, "done")
            self.assertEqual(owner._number_samples_per_class["number_classes"], 200)
            self.assertEqual(len(owner.list_folds), 2)


if __name__ == "__main__":
    unittest.main()
