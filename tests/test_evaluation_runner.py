import unittest
from types import SimpleNamespace

import numpy

from Engine.Evaluation.EvaluationRunner import EvaluationDataset
from Engine.Evaluation.EvaluationRunner import EvaluationMode
from Engine.Evaluation.EvaluationRunner import EvaluationRunner


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


if __name__ == "__main__":
    unittest.main()
