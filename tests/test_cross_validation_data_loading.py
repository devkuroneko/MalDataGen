import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.Evaluation.CrossValidation import _apply_bundle_to_owner
from Engine.Evaluation.CrossValidation import _apply_stratified_split_selection
from Engine.Evaluation.CrossValidation import _build_provided_split_folds
from Engine.Evaluation.CrossValidation import StratifiedData
from Engine.Evaluation.CrossValidation import load_dataset_from_args
from Engine.DataIO.DatasetContracts import SplitData


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
        execution_mode="normal",
        max_train_samples=None,
        max_samples_per_class=None,
        train_samples_per_class=None,
        test_samples_per_class=None,
        min_samples_per_class_required=1,
        strict_min_samples_per_class=False,
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

    def test_provided_split_keeps_valid_for_internal_evaluation_but_requires_test_for_final(self):
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
            self.assertTrue(fold["evaluation_not_applicable"])
            self.assertEqual(fold["validation_split_name"], "valid")
            self.assertIsNone(fold["real_test_split"])
            self.assertEqual(fold["training_source_indices"].tolist(), [0, 1, 2])
            self.assertEqual(fold["evaluation_source_indices"].tolist(), [0, 1])

    def test_provided_split_uses_test_as_final_real_split_when_valid_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(directory, "train", numpy.zeros((3, 2)), [0, 1, 2])
            self._save_split(directory, "valid", numpy.ones((2, 2)), [1, 2])
            self._save_split(directory, "test", numpy.full((4, 2), 2.0), [0, 0, 1, 2])

            args = npy_args(directory, split_mode="provided", with_valid=True, with_test=True)
            bundle = load_dataset_from_args(args)
            owner = SimpleNamespace(arguments=args, list_folds=[])

            _apply_bundle_to_owner(owner, bundle)
            _build_provided_split_folds(owner, bundle)

            self.assertEqual(len(owner.list_folds), 1)
            fold = owner.list_folds[0]
            self.assertEqual(fold["evaluation_split_name"], "test")
            self.assertFalse(fold["evaluation_not_applicable"])
            self.assertEqual(fold["x_evaluation_real"].shape, (4, 2))
            self.assertEqual(fold["real_train_split"].name, "train")
            self.assertEqual(fold["real_valid_split"].name, "valid")
            self.assertEqual(fold["real_test_split"].name, "test")
            self.assertEqual(fold["split_metadata"]["test"]["y_path"], str(Path(directory) / "test_y.npy"))
            self.assertEqual(fold["split_metadata"]["test"]["minimum_class_count"], 1)

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
            for fold in owner.list_folds:
                self.assertEqual(fold["x_training_real"].shape[0], fold["y_training_real"].shape[0])
                self.assertEqual(fold["x_evaluation_real"].shape[0], fold["y_evaluation_real"].shape[0])
                self.assertEqual(fold["training_source_indices"].shape[0], fold["y_training_real"].shape[0])
                self.assertEqual(fold["evaluation_source_indices"].shape[0], fold["y_evaluation_real"].shape[0])

    def test_batches_provided_split_uses_stratified_selection_for_train_and_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(
                directory,
                "train",
                numpy.arange(18).reshape(9, 2),
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
            )
            self._save_split(
                directory,
                "valid",
                numpy.arange(12).reshape(6, 2),
                [0, 0, 1, 1, 2, 2],
            )

            args = npy_args(directory, split_mode="provided", with_valid=True)
            args.execution_mode = "batches"
            args.num_classes = 3
            args.train_samples_per_class = 2
            args.test_samples_per_class = 1
            owner = SimpleNamespace(arguments=args, list_folds=[], current_subdir=directory)
            bundle = load_dataset_from_args(args)

            _apply_bundle_to_owner(owner, bundle)
            _build_provided_split_folds(owner, bundle)

            train_counts = dict(zip(*numpy.unique(owner.list_folds[0]["y_training_real"], return_counts=True)))
            valid_counts = dict(zip(*numpy.unique(owner.list_folds[0]["y_evaluation_real"], return_counts=True)))
            self.assertEqual(train_counts, {0: 2, 1: 2, 2: 2})
            self.assertEqual(valid_counts, {0: 1, 1: 1, 2: 1})
            self.assertTrue((Path(directory) / "SelectionReports" / "train_stratified_selection.json").is_file())
            self.assertTrue((Path(directory) / "SelectionReports" / "valid_stratified_selection.json").is_file())

    def test_batches_provided_test_quota_uses_test_not_low_coverage_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save_split(
                directory,
                "train",
                numpy.arange(18).reshape(9, 2),
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
            )
            self._save_split(
                directory,
                "valid",
                numpy.arange(6).reshape(3, 2),
                [0, 1, 2],
            )
            self._save_split(
                directory,
                "test",
                numpy.arange(18).reshape(9, 2),
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
            )

            args = npy_args(directory, split_mode="provided", with_valid=True, with_test=True)
            args.execution_mode = "batches"
            args.num_classes = 3
            args.train_samples_per_class = 2
            args.test_samples_per_class = 2
            args.strict_min_samples_per_class = True
            owner = SimpleNamespace(arguments=args, list_folds=[], current_subdir=directory)
            bundle = load_dataset_from_args(args)

            _apply_bundle_to_owner(owner, bundle)
            _build_provided_split_folds(owner, bundle)

            evaluation_counts = dict(zip(*numpy.unique(owner.list_folds[0]["y_evaluation_real"], return_counts=True)))
            valid_counts = dict(zip(*numpy.unique(bundle.valid.y, return_counts=True)))
            test_counts = dict(zip(*numpy.unique(bundle.test.y, return_counts=True)))
            self.assertEqual(evaluation_counts, {0: 2, 1: 2, 2: 2})
            self.assertEqual(valid_counts, {0: 1, 1: 1, 2: 1})
            self.assertEqual(test_counts, {0: 2, 1: 2, 2: 2})
            self.assertEqual(owner.list_folds[0]["real_test_split"].name, "test")

    def test_batches_selection_rejects_y_path_from_different_split_before_indexing(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_y_path = Path(directory) / "wrong_y.npy"
            numpy.save(wrong_y_path, numpy.array([0, 0, 1, 1, 2, 2], dtype=numpy.int64))
            owner = SimpleNamespace(
                arguments=SimpleNamespace(
                    num_classes=3,
                    min_samples_per_class_required=1,
                    strict_min_samples_per_class=False,
                    mmap_npy=True,
                ),
                current_subdir=directory,
                _dataset_bundle=SimpleNamespace(schema=SimpleNamespace(num_classes=3)),
            )
            split = SplitData(
                X=numpy.zeros((3, 2), dtype=numpy.float32),
                y=numpy.array([0, 1, 2], dtype=numpy.int64),
                name="valid",
            )

            with self.assertRaisesRegex(ValueError, "y_path row mismatch.*split=valid"):
                _apply_stratified_split_selection(owner, split, wrong_y_path, "valid", 1)


if __name__ == "__main__":
    unittest.main()
