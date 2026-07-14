import unittest
from types import SimpleNamespace

import numpy

from Engine.Evaluation.TrTs import TrTs
from Engine.Evaluation.TsTr import TsTr


class ConstantClassifier:
    def predict(self, data):
        return numpy.zeros(len(data), dtype=int)


class EvaluationLabelShapeOwner(TrTs, TsTr):
    def __init__(self, arguments=None):
        self.arguments = arguments or SimpleNamespace(
            evaluation_protocol="legacy",
            source_profile="legacy_csv",
        )
        self.data_generated = {0: numpy.zeros((2, 2), dtype=numpy.float32)}
        self._dictionary_classifiers_name = ["Dummy"]
        self._labels_are_discrete = True
        self.fold_number = 0
        self.metrics_calls = []
        self.distance_calls = []
        self._dictionary_metrics = {"EvaluationMetadata": {}, "Diagnostics": {}}

    def get_trained_classifiers(self, x_values, y_values, dtype, number_columns):
        self.trained_x = numpy.asarray(x_values)
        self.trained_y_shape = numpy.asarray(y_values).shape
        self.trained_y = numpy.asarray(y_values)
        return [ConstantClassifier()]

    def get_number_columns(self):
        return 2

    def get_task_metrics(self, y_true, y_pred, evaluation_name, classifier_name, fold_number):
        self.metrics_calls.append((
            numpy.asarray(y_true).shape,
            numpy.asarray(y_pred).shape,
            evaluation_name,
            classifier_name,
            fold_number,
        ))

    def get_distance_metrics(self, data_real, data_synthetic, evaluation_name, fold_number):
        self.distance_calls.append((evaluation_name, fold_number))

    def mark_evaluation_classifiers_not_applicable(self, *args, **kwargs):
        raise AssertionError("evaluation should be applicable in this regression test")

    def mark_distance_metrics_not_applicable(self, *args, **kwargs):
        raise AssertionError("distance metrics should be applicable in this regression test")


class EvaluationLabelShapeTest(unittest.TestCase):

    def test_tr_ts_accepts_one_dimensional_labels(self):
        owner = EvaluationLabelShapeOwner()
        dictionary_data = {
            "x_evaluation_real": numpy.zeros((2, 2), dtype=numpy.float32),
            "y_evaluation_real": numpy.array([0, 0], dtype=int),
        }

        owner.evaluation_TR_TS(dictionary_data, owner.data_generated)

        self.assertEqual(owner.trained_y_shape, (2,))
        self.assertEqual(owner.metrics_calls[0][0], (2,))

    def test_tr_ts_strict_protocol_trains_on_real_training_split(self):
        owner = EvaluationLabelShapeOwner(SimpleNamespace(
            evaluation_protocol="appclassnet_strict",
            source_profile="appclassnet_top200",
        ))
        dictionary_data = {
            "x_training_real": numpy.ones((3, 2), dtype=numpy.float32),
            "y_training_real": numpy.array([0, 1, 1], dtype=int),
            "x_evaluation_real": numpy.zeros((2, 2), dtype=numpy.float32),
            "y_evaluation_real": numpy.array([0, 0], dtype=int),
        }

        owner.evaluation_TR_TS(dictionary_data, owner.data_generated)

        self.assertEqual(owner.trained_x.shape, (3, 2))
        self.assertEqual(owner.trained_y.tolist(), [0, 1, 1])
        metadata = owner._dictionary_metrics["EvaluationMetadata"]["1-Fold"]["TR-TS"]
        self.assertEqual(metadata["train_origin"], "real:training")

    def test_ts_tr_accepts_one_dimensional_labels(self):
        owner = EvaluationLabelShapeOwner()
        dictionary_data = {
            "x_training_real": numpy.zeros((2, 2), dtype=numpy.float32),
            "x_evaluation_real": numpy.zeros((2, 2), dtype=numpy.float32),
            "y_evaluation_real": numpy.array([0, 0], dtype=int),
        }

        owner.evaluation_TS_TR(dictionary_data, owner.data_generated)

        self.assertEqual(owner.metrics_calls[0][0], (2,))
        self.assertEqual(owner.distance_calls, [("R-S", 1)])

    def test_ts_tr_normal_metrics_use_full_real_evaluation_labels(self):
        owner = EvaluationLabelShapeOwner()
        owner.data_generated = {0: numpy.zeros((1, 2), dtype=numpy.float32)}
        dictionary_data = {
            "x_training_real": numpy.zeros((3, 2), dtype=numpy.float32),
            "x_evaluation_real": numpy.zeros((3, 2), dtype=numpy.float32),
            "y_evaluation_real": numpy.array([0, 0, 0], dtype=int),
        }

        owner.evaluation_TS_TR(dictionary_data, owner.data_generated)

        self.assertEqual(owner.metrics_calls[0][0], (3,))
        self.assertEqual(owner.metrics_calls[0][1], (3,))


if __name__ == "__main__":
    unittest.main()
