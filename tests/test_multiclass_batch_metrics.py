import tempfile
import unittest
from pathlib import Path

import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from Engine.DataIO.JsonIO import load_json_file
from Engine.Evaluation.MulticlassBatchMetrics import MulticlassConfusionAccumulator
from Engine.Evaluation.MulticlassBatchMetrics import evaluate_classifier_batchwise
from Engine.Evaluation.TrTrPipeline import TrTrConfig
from Engine.Evaluation.TrTrPipeline import run_tr_tr_pipeline
from tests.test_tr_tr_pipeline import make_bundle


class SequenceClassifier:
    def __init__(self, predictions):
        self.predictions = numpy.asarray(predictions, dtype=numpy.int64)
        self.offset = 0

    def predict(self, x_values):
        count = int(x_values.shape[0])
        result = self.predictions[self.offset:self.offset + count]
        self.offset += count
        return result


class MulticlassBatchMetricsTest(unittest.TestCase):

    def test_metrics_match_sklearn_on_small_dataset(self):
        labels = [0, 1, 2]
        y_true = numpy.asarray([0, 0, 1, 1, 1, 2, 2], dtype=numpy.int64)
        y_pred = numpy.asarray([0, 1, 1, 0, 1, 2, 0], dtype=numpy.int64)
        accumulator = MulticlassConfusionAccumulator(labels)
        accumulator.update(y_true, y_pred)

        metrics = accumulator.metrics(batch_size=10, inference_time_seconds=0.5)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        self.assertTrue(numpy.array_equal(
            accumulator.confusion_matrix,
            confusion_matrix(y_true, y_pred, labels=labels),
        ))
        self.assertAlmostEqual(metrics["accuracy"], accuracy_score(y_true, y_pred))
        self.assertAlmostEqual(metrics["balanced_accuracy"], balanced_accuracy_score(y_true, y_pred))
        self.assertAlmostEqual(metrics["macro_precision"], precision)
        self.assertAlmostEqual(metrics["macro_recall"], recall)
        self.assertAlmostEqual(metrics["macro_f1"], f1)
        self.assertAlmostEqual(metrics["weighted_precision"], weighted_precision)
        self.assertAlmostEqual(metrics["weighted_recall"], weighted_recall)
        self.assertAlmostEqual(metrics["weighted_f1"], weighted_f1)

    def test_matrix_accumulated_in_batches_equals_one_shot(self):
        labels = [0, 1, 2, 3]
        y_true = numpy.asarray([0, 1, 2, 3, 0, 1, 2, 3, 3], dtype=numpy.int64)
        y_pred = numpy.asarray([0, 1, 1, 3, 2, 1, 2, 0, 3], dtype=numpy.int64)
        one_shot = MulticlassConfusionAccumulator(labels)
        one_shot.update(y_true, y_pred)
        batched = MulticlassConfusionAccumulator(labels)
        for start in range(0, y_true.shape[0], 3):
            batched.update(y_true[start:start + 3], y_pred[start:start + 3])

        self.assertTrue(numpy.array_equal(batched.confusion_matrix, one_shot.confusion_matrix))
        self.assertEqual(batched.batch_count, 3)

    def test_supports_200_classes(self):
        labels = list(range(200))
        y_true = numpy.arange(200, dtype=numpy.int64)
        y_pred = numpy.arange(200, dtype=numpy.int64)
        accumulator = MulticlassConfusionAccumulator(labels)
        accumulator.update(y_true, y_pred)

        metrics = accumulator.metrics()

        self.assertEqual(accumulator.confusion_matrix.shape, (200, 200))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_records_class_without_prediction(self):
        accumulator = MulticlassConfusionAccumulator([0, 1, 2])
        accumulator.update([0, 1, 1, 2], [0, 0, 0, 2])

        metrics = accumulator.metrics()

        self.assertIn(1, metrics["classes_without_predictions"])
        self.assertEqual(metrics["predicted_count_by_class"]["1"], 0)

    def test_records_class_without_support_with_zero_division_zero(self):
        accumulator = MulticlassConfusionAccumulator([0, 1, 2])
        accumulator.update([0, 1], [0, 1])

        metrics = accumulator.metrics()
        rows = {row["class_id"]: row for row in accumulator.per_class_rows()}

        self.assertIn(2, metrics["classes_without_support"])
        self.assertEqual(rows[2]["support"], 0)
        self.assertEqual(rows[2]["recall"], 0.0)
        self.assertEqual(rows[2]["precision"], 0.0)
        self.assertEqual(rows[2]["f1"], 0.0)

    def test_last_batch_smaller_than_batch_size(self):
        x_values = numpy.arange(30, dtype=numpy.float32).reshape(10, 3)
        y_true = numpy.asarray([0, 1] * 5, dtype=numpy.int64)
        classifier = SequenceClassifier(y_true)

        accumulator, stats = evaluate_classifier_batchwise(
            classifier,
            x_values,
            y_true,
            class_labels=[0, 1],
            batch_size=4,
        )

        self.assertEqual(stats["batch_count"], 3)
        self.assertEqual(stats["total_predictions"], 10)
        self.assertEqual(accumulator.metrics()["accuracy"], 1.0)

    def test_global_y_pred_is_not_required_in_batchwise_evaluation(self):
        x_values = numpy.arange(12, dtype=numpy.float32).reshape(6, 2)
        y_true = numpy.asarray([0, 1, 0, 1, 0, 1], dtype=numpy.int64)
        classifier = SequenceClassifier(y_true)

        accumulator, stats = evaluate_classifier_batchwise(
            classifier,
            x_values,
            y_true,
            class_labels=[0, 1],
            batch_size=2,
        )

        self.assertFalse(hasattr(accumulator, "y_pred"))
        self.assertEqual(stats["total_predictions"], 6)

    def test_weighted_recall_matches_expected_accuracy(self):
        accumulator = MulticlassConfusionAccumulator([0, 1])
        accumulator.update([0, 0, 1, 1, 1], [0, 1, 1, 0, 1])

        metrics = accumulator.metrics()

        self.assertAlmostEqual(metrics["weighted_recall"], 3.0 / 5.0)
        self.assertAlmostEqual(metrics["weighted_recall"], metrics["accuracy"])

    def test_artifacts_are_written(self):
        accumulator = MulticlassConfusionAccumulator([0, 1, 2])
        accumulator.update([0, 1, 2], [0, 1, 1])
        metrics = accumulator.metrics(batch_size=2, inference_time_seconds=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = accumulator.save_artifacts(
                tmpdir,
                {"protocol": "TR-TR", "status": "completed", **metrics},
            )

            self.assertTrue(Path(artifacts["metrics"]).is_file())
            self.assertTrue(Path(artifacts["confusion_matrix"]).is_file())
            self.assertTrue(Path(artifacts["per_class_metrics"]).is_file())
            saved_matrix = numpy.load(artifacts["confusion_matrix"])
            saved_metrics = load_json_file(artifacts["metrics"])

        self.assertEqual(saved_matrix.shape, (3, 3))
        self.assertEqual(saved_metrics["batch_count"], 1)

    def test_tr_tr_writes_batch_metrics_artifacts_without_predictions(self):
        bundle = make_bundle(num_classes=3, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom", eval_batch_size=5),
                tmpdir,
            )
            metrics = load_json_file(Path(result["artifacts"]["metrics"]))
            confusion = numpy.load(result["artifacts"]["confusion_matrix"])
            per_class_text = Path(result["artifacts"]["per_class_metrics"]).read_text(encoding="utf-8")

        self.assertEqual(result["protocol"], "TR-TR")
        self.assertEqual(result["evaluation"]["batch_count"], 3)
        self.assertEqual(metrics["batch_count"], 3)
        self.assertEqual(metrics["total_predictions"], 12)
        self.assertEqual(confusion.shape, (3, 3))
        self.assertNotIn("y_pred", result)
        self.assertIn("class_id,support,predicted_count,true_positive,precision,recall,f1", per_class_text)


if __name__ == "__main__":
    unittest.main()
