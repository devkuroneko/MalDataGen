import unittest
from types import SimpleNamespace

from Engine.Metrics.Metrics import Metrics
from Engine.Metrics.Metrics import NOT_APPLICABLE


class ConstantMetric:
    def __init__(self, value):
        self.value = value

    def get_metric(self, real_labels, predict_labels):
        return self.value


class MetricsTargetTypesTest(unittest.TestCase):

    def _metrics(self, target_type, metric_names=None):
        metrics = Metrics.__new__(Metrics)
        metrics._target_type = target_type
        metrics._data_type = "binary"
        metrics.arguments = SimpleNamespace(num_classes=None, number_samples_per_class=None)
        metrics._dictionary_classifiers_name = ["DummyClassifier"]
        metrics._dictionary_binary_metrics = {
            "Accuracy": ConstantMetric(1.0),
            "Precision": ConstantMetric(0.0),
        }
        metrics.list_classifier_metrics = metric_names or metrics._get_classifier_metric_names()
        metrics._dictionary_metrics = {
            "TR-TS": {
                "DummyClassifier": {
                    "1-Fold": {metric: NOT_APPLICABLE for metric in metrics.list_classifier_metrics},
                    "Summary": {
                        metric: {"mean": NOT_APPLICABLE, "std": NOT_APPLICABLE}
                        for metric in metrics.list_classifier_metrics
                    },
                }
            },
            "TS-TR": {"DummyClassifier": {"1-Fold": {}, "Summary": {}}},
            "TR-TR": {"DummyClassifier": {"1-Fold": {}, "Summary": {}}},
            "DistanceMetrics": {"R-S": {"1-Fold": {}, "Summary": {}}, "R-R": {"1-Fold": {}, "Summary": {}}},
            "Diagnostics": {"1-Fold": {}, "Summary": {}},
        }
        return metrics

    def test_binary_metrics_keep_zero_as_real_value(self):
        metrics = self._metrics("binary")

        metrics.get_task_metrics([0, 1], [0, 1], "TR-TS", "DummyClassifier", 1)

        fold = metrics._dictionary_metrics["TR-TS"]["DummyClassifier"]["1-Fold"]
        self.assertEqual(fold["Accuracy"], 1.0)
        self.assertEqual(fold["Precision"], 0.0)

    def test_multiclass_three_classes_metrics(self):
        metrics = self._metrics("multiclass")

        metrics.get_task_metrics([0, 1, 2], [0, 2, 2], "TR-TS", "DummyClassifier", 1)

        fold = metrics._dictionary_metrics["TR-TS"]["DummyClassifier"]["1-Fold"]
        self.assertAlmostEqual(fold["Accuracy"], 2 / 3)
        self.assertIn("MacroPrecision", fold)
        self.assertIn("MacroRecall", fold)
        self.assertIn("MacroF1", fold)
        self.assertIn("WeightedPrecision", fold)
        self.assertIn("WeightedRecall", fold)
        self.assertIn("WeightedF1", fold)
        self.assertIn("BalancedAccuracy", fold)

    def test_multiclass_missing_synthetic_class_is_numeric_not_fake_skip(self):
        metrics = self._metrics("multiclass")

        metrics.get_task_metrics([0, 1, 2], [0, 0, 0], "TR-TS", "DummyClassifier", 1)

        fold = metrics._dictionary_metrics["TR-TS"]["DummyClassifier"]["1-Fold"]
        self.assertEqual(fold["Accuracy"], 1 / 3)
        self.assertNotEqual(fold["MacroF1"], NOT_APPLICABLE)
        self.assertNotEqual(fold["BalancedAccuracy"], NOT_APPLICABLE)

    def test_regression_classification_metrics_are_not_applicable(self):
        metrics = self._metrics("regression")

        metrics.get_task_metrics([0.1, 0.2], [0.1, 0.3], "TR-TS", "DummyClassifier", 1)

        fold = metrics._dictionary_metrics["TR-TS"]["DummyClassifier"]["1-Fold"]
        self.assertTrue(all(value == NOT_APPLICABLE for value in fold.values()))
        self.assertIn("TR-TS:DummyClassifier", metrics._dictionary_metrics["NotApplicable"]["1-Fold"])

    def test_multiclass_chance_level_suspected_for_top_200(self):
        metrics = self._metrics("multiclass")
        metrics.arguments = SimpleNamespace(num_classes=200, number_samples_per_class=None)

        metrics.get_task_metrics(
            list(range(200)),
            [0] * 200,
            "TR-TS",
            "DummyClassifier",
            1,
        )

        diagnostics = metrics._dictionary_metrics["Diagnostics"]["1-Fold"]["TR-TS"]["DummyClassifier"]
        self.assertTrue(diagnostics["chance_level_suspected"])
        self.assertEqual(diagnostics["num_classes"], 200)


if __name__ == "__main__":
    unittest.main()
