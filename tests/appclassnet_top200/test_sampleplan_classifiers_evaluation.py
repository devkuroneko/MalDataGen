import numpy
import pytest
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from Engine.DataIO.RealClassCountPolicy import build_split_sample_plan
from Engine.Evaluation.ClassifierFactory import ClassifierConfig
from Engine.Evaluation.ClassifierFactory import build_classifier
from Engine.Evaluation.ClassifierFactory import estimator_statistics
from Engine.Evaluation.MulticlassBatchMetrics import MulticlassConfusionAccumulator
from Engine.Evaluation.MulticlassBatchMetrics import evaluate_classifier_batchwise


def test_sampleplan_all_has_no_selected_indices(small_bundle):
    plan = build_split_sample_plan(
        small_bundle.train.y,
        None,
        4,
        7,
        "train",
        strategy="all",
    )

    assert plan.selected_indices is None, "strategy=all must not materialize subset indices"
    assert plan.total_rows == small_bundle.train.num_rows


def test_sampleplan_balanced_per_class_deterministic(small_bundle):
    first = build_split_sample_plan(small_bundle.train.y, 2, 4, 11, "train", strategy="balanced_per_class")
    second = build_split_sample_plan(small_bundle.train.y, 2, 4, 11, "train", strategy="balanced_per_class")

    numpy.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    assert numpy.max(first.selected_indices) < small_bundle.train.num_rows
    assert numpy.min(first.selected_indices) >= 0


def test_sampleplan_random_state_can_change_selection(small_bundle):
    first = build_split_sample_plan(small_bundle.train.y, 2, 4, 11, "train", strategy="balanced_per_class")
    second = build_split_sample_plan(small_bundle.train.y, 2, 4, 12, "train", strategy="balanced_per_class")

    assert not numpy.array_equal(first.selected_indices, second.selected_indices), "different seeds should affect subset order"


def test_sampleplan_up_to_available_caps_small_class(imbalanced_labels):
    plan = build_split_sample_plan(
        imbalanced_labels,
        500,
        3,
        3,
        "train",
        strategy="up_to_available",
    )

    row_for_class_1 = next(row for row in plan.selection_table if row["class_id"] == 1)
    assert row_for_class_1["available"] == 499, "fixture must reproduce class below 500"
    assert row_for_class_1["selected"] == 499, "up_to_available must cap class below 500"


def test_sampleplan_balanced_strict_fails_for_class_below_500(imbalanced_labels):
    with pytest.raises(ValueError, match="fewer than requested 500"):
        build_split_sample_plan(
            imbalanced_labels,
            500,
            3,
            3,
            "train",
            strategy="balanced_per_class",
            insufficient_policy="strict",
        )


def test_sampleplan_reproduces_out_of_bounds_case_before_indexing():
    labels = numpy.asarray([0] * 40916 + [1] * 40916, dtype=numpy.int64)
    plan = build_split_sample_plan(labels, 100, 2, 0, "test", strategy="balanced_per_class")
    bad_indices = numpy.asarray([98070], dtype=numpy.int64)

    assert labels.shape[0] == 81832
    assert numpy.max(plan.selected_indices) < 81832, "valid planner must stay inside current split"
    assert int(bad_indices.max()) >= labels.shape[0], "regression case must be out of bounds before indexing"


def test_classifier_factory_builds_decision_tree_with_parameters():
    model = build_classifier(
        ClassifierConfig(
            classifier="decision_tree",
            random_state=17,
            decision_tree_max_depth=3,
            decision_tree_criterion="entropy",
        )
    )

    params = model.get_params(deep=True)
    assert model.__class__.__name__ == "DecisionTreeClassifier"
    assert params["random_state"] == 17
    assert params["max_depth"] == 3
    assert params["criterion"] == "entropy"
    assert not hasattr(model, "partial_fit"), "DecisionTreeClassifier must not be treated as incremental"


def test_classifier_factory_builds_random_forest_with_parameters():
    model = build_classifier(
        ClassifierConfig(
            classifier="random_forest",
            random_state=19,
            random_forest_n_estimators=5,
            random_forest_n_jobs=1,
            random_forest_max_depth=4,
        )
    )

    params = model.get_params(deep=True)
    assert model.__class__.__name__ == "RandomForestClassifier"
    assert params["n_estimators"] == 5
    assert params["n_jobs"] == 1
    assert params["max_depth"] == 4
    assert not hasattr(model, "partial_fit"), "RandomForestClassifier must not be treated as incremental"


def test_classifier_statistics_for_tree_and_forest(small_bundle):
    tree = build_classifier(ClassifierConfig(classifier="decision_tree", random_state=1))
    tree.fit(small_bundle.train.X, small_bundle.train.y)
    tree_stats = estimator_statistics(tree, "decision_tree")

    forest = build_classifier(ClassifierConfig(classifier="random_forest", random_state=1, random_forest_n_estimators=3))
    forest.fit(small_bundle.train.X, small_bundle.train.y)
    forest_stats = estimator_statistics(forest, "random_forest")

    assert tree_stats["node_count"] > 0
    assert tree_stats["leaf_count"] > 0
    assert forest_stats["n_estimators_trained"] == 3
    assert forest_stats["total_nodes"] > 0


def test_unknown_or_subset_classifier_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported classifier"):
        ClassifierConfig(classifier="unknown")
    with pytest.raises(ValueError, match="subset classifier"):
        ClassifierConfig(classifier="decision_tree_subset")


def test_incremental_confusion_metrics_match_sklearn():
    y_true = numpy.asarray([0, 1, 2, 2, 1, 0, 3, 3], dtype=numpy.int64)
    y_pred = numpy.asarray([0, 2, 2, 1, 1, 0, 0, 3], dtype=numpy.int64)
    accumulator = MulticlassConfusionAccumulator(class_labels=[0, 1, 2, 3])
    accumulator.update(y_true[:3], y_pred[:3])
    accumulator.update(y_true[3:], y_pred[3:])
    metrics = accumulator.metrics(batch_size=3, inference_time_seconds=0.5)

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert metrics["macro_precision"] == pytest.approx(precision_score(y_true, y_pred, average="macro", zero_division=0))
    assert metrics["macro_recall"] == pytest.approx(recall_score(y_true, y_pred, average="macro", zero_division=0))
    assert metrics["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro", zero_division=0))


def test_incremental_confusion_matrix_works_for_200_classes():
    accumulator = MulticlassConfusionAccumulator(class_labels=list(range(200)))
    y_true = numpy.arange(200, dtype=numpy.int64)
    y_pred = numpy.arange(200, dtype=numpy.int64)
    accumulator.update(y_true[:123], y_pred[:123])
    accumulator.update(y_true[123:], y_pred[123:])

    assert accumulator.confusion_matrix.shape == (200, 200)
    assert int(numpy.trace(accumulator.confusion_matrix)) == 200


class ConstantBatchClassifier:
    def fit(self, x_values, y_values):
        return self

    def predict(self, x_values):
        return numpy.zeros(int(x_values.shape[0]), dtype=numpy.int64)


def test_batch_evaluation_handles_incomplete_last_batch(top200_bundle):
    classifier = ConstantBatchClassifier()
    accumulator, stats = evaluate_classifier_batchwise(
        classifier,
        top200_bundle.test.X[:200],
        top200_bundle.test.y[:200],
        class_labels=list(range(200)),
        batch_size=64,
    )
    metrics = accumulator.metrics(batch_size=64, inference_time_seconds=stats["inference_time_seconds"])

    assert metrics["batch_count"] == 4, "200 samples with batch size 64 must produce incomplete fourth batch"
    assert metrics["total_predictions"] == 200
    assert 199 in metrics["classes_without_predictions"], "constant classifier should not predict class 199"
