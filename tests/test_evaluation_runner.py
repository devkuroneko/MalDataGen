import unittest
from types import SimpleNamespace

import numpy

from Engine.Evaluation.EvaluationRunner import EvaluationDataset
from Engine.Evaluation.EvaluationRunner import EvaluationMode
from Engine.Evaluation.EvaluationRunner import EvaluationRunner
from Engine.Evaluation.EvaluationRunner import split_to_dictionary
from Engine.DataIO.DatasetContracts import SplitData


class EvaluationRunnerOwner:
    def __init__(self):
        self.arguments = SimpleNamespace(
            evaluation_protocol="appclassnet_strict",
            source_profile="appclassnet_top200",
        )
        self._dictionary_metrics = {"EvaluationMetadata": {}}

    def get_trained_classifiers(self, *args, **kwargs):
        raise AssertionError("not needed")

    def get_number_columns(self):
        return 2


class EvaluationRunnerTest(unittest.TestCase):

    def test_save_results_records_hashes_origins_and_class_counts(self):
        owner = EvaluationRunnerOwner()
        runner = EvaluationRunner(owner, EvaluationMode.TR_TR)
        dataset = EvaluationDataset(
            X_train=numpy.array([[0.0, 0.1], [0.2, 0.3]], dtype=numpy.float32),
            y_train=numpy.array([0, 1], dtype=numpy.int64),
            X_test=numpy.array([[0.0, 0.1]], dtype=numpy.float32),
            y_test=numpy.array([0], dtype=numpy.int64),
            train_origin="real:training",
            test_origin="real:evaluation",
        )

        runner.validate(dataset)
        runner.save_results(dataset, 1)

        metadata = owner._dictionary_metrics["EvaluationMetadata"]["1-Fold"]["TR-TR"]
        self.assertEqual(metadata["schema_version"], "evaluation_dataset/v1")
        self.assertEqual(metadata["train_origin"], "real:training")
        self.assertEqual(metadata["test_origin"], "real:evaluation")
        self.assertEqual(metadata["train_class_counts"], {"0": 1, "1": 1})
        self.assertEqual(metadata["test_class_counts"], {"0": 1})
        self.assertEqual(len(metadata["train_hash"]), 64)
        self.assertEqual(len(metadata["test_hash"]), 64)

    def test_build_dataset_records_provided_split_names_and_paths(self):
        owner = EvaluationRunnerOwner()
        runner = EvaluationRunner(owner, EvaluationMode.TR_TR)
        dictionary_data = split_to_dictionary(
            real_train_data=SplitData(
                X=numpy.zeros((4, 2), dtype=numpy.float32),
                y=numpy.array([0, 0, 1, 1]),
                name="train",
                x_path="/dataset/train_x.npy",
                y_path="/dataset/train_y.npy",
                dataset_id="appclassnet_top200",
            ),
            real_test_data=SplitData(
                X=numpy.ones((6, 2), dtype=numpy.float32),
                y=numpy.array([0, 0, 0, 1, 1, 1]),
                name="test",
                x_path="/dataset/test_x.npy",
                y_path="/dataset/test_y.npy",
                dataset_id="appclassnet_top200",
            ),
        )

        dataset = runner.build_evaluation_dataset(dictionary_data)
        runner.validate(dataset)
        runner.save_results(dataset, 1)

        metadata = owner._dictionary_metrics["EvaluationMetadata"]["1-Fold"]["TR-TR"]
        self.assertEqual(metadata["train_split_name"], "train")
        self.assertEqual(metadata["test_split_name"], "test")
        self.assertEqual(metadata["train_x_path"], "/dataset/train_x.npy")
        self.assertEqual(metadata["test_y_path"], "/dataset/test_y.npy")
        self.assertEqual(metadata["test_minimum_class_count"], 3)

    def test_validate_rejects_misaligned_train_labels(self):
        owner = EvaluationRunnerOwner()
        runner = EvaluationRunner(owner, EvaluationMode.TR_TR)
        dataset = EvaluationDataset(
            X_train=numpy.zeros((2, 2), dtype=numpy.float32),
            y_train=numpy.array([0], dtype=numpy.int64),
            X_test=numpy.zeros((1, 2), dtype=numpy.float32),
            y_test=numpy.array([0], dtype=numpy.int64),
            train_origin="real:training",
            test_origin="real:evaluation",
        )

        with self.assertRaisesRegex(ValueError, "train X/y length mismatch"):
            runner.validate(dataset)

    def test_tr_ts_tr_builds_augmented_train_and_real_test_dataset(self):
        owner = EvaluationRunnerOwner()
        runner = EvaluationRunner(owner, EvaluationMode.TR_TS_TR)
        dictionary_data = split_to_dictionary(
            real_train_data=SplitData(
                X=numpy.array([[0.0, 0.1], [0.2, 0.3]], dtype=numpy.float32),
                y=numpy.array([0, 1]),
                name="train",
            ),
            real_test_data=SplitData(
                X=numpy.array([[0.4, 0.5], [0.6, 0.7]], dtype=numpy.float32),
                y=numpy.array([0, 1]),
                name="test",
            ),
        )
        synthetic = {
            0: numpy.array([[0.01, 0.11]], dtype=numpy.float32),
            1: numpy.array([[0.21, 0.31]], dtype=numpy.float32),
        }

        dataset = runner.build_evaluation_dataset(dictionary_data, synthetic)
        runner.validate(dataset)
        runner.save_results(dataset, 1)

        self.assertEqual(dataset.X_train.shape, (4, 2))
        self.assertEqual(dataset.y_train.tolist(), [0, 1, 0, 1])
        self.assertEqual(dataset.test_origin, "real:test")
        metadata = owner._dictionary_metrics["EvaluationMetadata"]["1-Fold"]["TR+TS-TR"]
        self.assertEqual(metadata["train_class_counts"], {"0": 2, "1": 2})
        self.assertEqual(metadata["test_class_counts"], {"0": 1, "1": 1})


if __name__ == "__main__":
    unittest.main()
