import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData
from Engine.DataIO.JsonIO import load_json_file
from Engine.Evaluation.ClassifierFactory import ClassifierConfig
from Engine.Evaluation.ClassifierFactory import build_classifier
from Engine.Evaluation.TrTrPipeline import TrTrConfig
from Engine.Evaluation.TrTrPipeline import run_tr_tr_pipeline


def make_bundle(num_classes=2, rows_per_class=4, num_features=3, valid_marker=99.0):
    feature_names = [f"f{i}" for i in range(num_features)]
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    valid_x = []
    valid_y = []
    for class_id in range(num_classes):
        center = float(class_id * 10)
        for row in range(rows_per_class):
            train_x.append([center + row * 0.01 + feature for feature in range(num_features)])
            train_y.append(class_id)
            test_x.append([center + row * 0.02 + feature for feature in range(num_features)])
            test_y.append(class_id)
            valid_x.append([valid_marker + class_id + feature for feature in range(num_features)])
            valid_y.append(class_id)
    schema = DatasetSchema(
        feature_names=feature_names,
        num_features=num_features,
        feature_type="continuous",
        target_type="multiclass",
        num_classes=num_classes,
        classes=tuple(range(num_classes)),
        source_format="npy_xy",
        data_format="npy_xy",
        split_mode="provided",
        source_profile="custom",
        feature_dtype="float32",
        target_dtype="int64",
        data_space="source",
    )
    return DatasetBundle(
        train=SplitData(
            X=numpy.asarray(train_x, dtype=numpy.float32),
            y=numpy.asarray(train_y, dtype=numpy.int64),
            name="train",
            x_path="/dataset/train_x.npy",
            y_path="/dataset/train_y.npy",
        ),
        valid=SplitData(
            X=numpy.asarray(valid_x, dtype=numpy.float32),
            y=numpy.asarray(valid_y, dtype=numpy.int64),
            name="valid",
            x_path="/dataset/valid_x.npy",
            y_path="/dataset/valid_y.npy",
        ),
        test=SplitData(
            X=numpy.asarray(test_x, dtype=numpy.float32),
            y=numpy.asarray(test_y, dtype=numpy.int64),
            name="test",
            x_path="/dataset/test_x.npy",
            y_path="/dataset/test_y.npy",
        ),
        schema=schema,
    )


class SpyClassifier:
    def __init__(self):
        self.fit_x = None
        self.fit_y = None

    def fit(self, x_values, y_values):
        self.fit_x = numpy.asarray(x_values)
        self.fit_y = numpy.asarray(y_values)
        return self

    def predict(self, x_values):
        x_values = numpy.asarray(x_values)
        return numpy.where(x_values[:, 0] >= 5.0, 1, 0).astype(numpy.int64)

    def get_params(self):
        return {"spy": True}


class FailingFitClassifier:
    def fit(self, x_values, y_values):
        raise RuntimeError("fit failed for artifact test")

    def predict(self, x_values):  # pragma: no cover - fit always fails
        return numpy.zeros(int(x_values.shape[0]), dtype=numpy.int64)

    def get_params(self):
        return {"failing": True}


class TrTrPipelineTest(unittest.TestCase):

    def test_tr_tr_uses_only_train_real_and_test_real(self):
        bundle = make_bundle(num_classes=2, rows_per_class=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )

        self.assertEqual(result["protocol"], "TR-TR")
        self.assertEqual(result["source_train"], "real")
        self.assertEqual(result["source_test"], "real")
        self.assertEqual(result["files"]["train_x"], "/dataset/train_x.npy")
        self.assertEqual(result["files"]["test_y"], "/dataset/test_y.npy")
        self.assertEqual(result["valid_usage"], "ignored")
        self.assertEqual(result["status"], "completed")

    def test_generator_is_not_instantiated_or_imported(self):
        import sys
        module_name = "Engine.Models.GenerativeModels"
        sys.modules.pop(module_name, None)
        bundle = make_bundle(num_classes=2, rows_per_class=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )

        self.assertNotIn(module_name, sys.modules)

    def test_test_split_does_not_participate_in_fit(self):
        bundle = make_bundle(num_classes=2, rows_per_class=3)
        bundle.test.X[:, 0] = 777.0
        spy = SpyClassifier()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("Engine.Evaluation.TrTrPipeline.build_tr_tr_classifier", return_value=spy):
                run_tr_tr_pipeline(
                    bundle,
                    TrTrConfig(classifier="decision_tree", source_profile="custom"),
                    tmpdir,
                )

        self.assertIsNotNone(spy.fit_x)
        self.assertFalse(bool(numpy.any(spy.fit_x[:, 0] == 777.0)))

    def test_protocol_is_saved_as_tr_tr_and_not_not_run(self):
        bundle = make_bundle(num_classes=2, rows_per_class=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )
            saved = load_json_file(Path(result["artifacts"]["result"]))

        self.assertEqual(saved["protocol"], "TR-TR")
        self.assertEqual(saved["status"], "completed")
        self.assertNotEqual(saved["status"], "not_run")

    def test_execution_works_with_two_classes(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )

        self.assertEqual(result["num_classes"], 2)
        self.assertIn("Accuracy", result["metrics"])

    def test_execution_works_with_multiple_classes(self):
        bundle = make_bundle(num_classes=5, rows_per_class=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="random_forest", n_estimators=5, source_profile="custom"),
                tmpdir,
            )

        self.assertEqual(result["num_classes"], 5)
        self.assertEqual(result["classifier"], "random_forest")

    def test_execution_works_with_200_classes_small_dataset(self):
        bundle = make_bundle(num_classes=200, rows_per_class=1, num_features=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom", eval_batch_size=17),
                tmpdir,
            )

        self.assertEqual(result["num_classes"], 200)
        self.assertEqual(result["samples"]["train_total_effective"], 200)
        self.assertEqual(result["samples"]["test_total_effective"], 200)

    def test_all_and_balanced_subset_are_identified(self):
        bundle = make_bundle(num_classes=3, rows_per_class=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            full = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                Path(tmpdir) / "full",
            )
            subset = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(
                    classifier="decision_tree",
                    train_sampling="balanced_per_class",
                    test_sampling="balanced_per_class",
                    train_samples_per_class=2,
                    test_samples_per_class=2,
                    source_profile="custom",
                ),
                Path(tmpdir) / "subset",
            )

        self.assertEqual(full["sampling_mode"], "full_provided_split")
        self.assertEqual(subset["sampling_mode"], "balanced_subset")
        self.assertEqual(subset["samples"]["train_total_effective"], 6)
        self.assertFalse(subset["classifier_subset"])

    def test_valid_is_not_mixed_into_train(self):
        bundle = make_bundle(num_classes=2, rows_per_class=3, valid_marker=555.0)
        spy = SpyClassifier()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("Engine.Evaluation.TrTrPipeline.build_tr_tr_classifier", return_value=spy):
                result = run_tr_tr_pipeline(
                    bundle,
                    TrTrConfig(classifier="decision_tree", source_profile="custom"),
                    tmpdir,
                )

        self.assertEqual(result["valid_usage"], "ignored")
        self.assertFalse(bool(numpy.any(spy.fit_x[:, 0] >= 555.0)))

    def test_factory_builds_decision_tree(self):
        model = build_classifier(ClassifierConfig(classifier="decision_tree", random_state=13))

        self.assertEqual(model.__class__.__name__, "DecisionTreeClassifier")
        self.assertEqual(model.get_params(deep=True)["random_state"], 13)
        self.assertEqual(model.get_params(deep=True)["criterion"], "gini")

    def test_factory_builds_random_forest(self):
        model = build_classifier(
            ClassifierConfig(
                classifier="random_forest",
                random_state=17,
                random_forest_n_estimators=7,
                random_forest_n_jobs=2,
            )
        )

        params = model.get_params(deep=True)
        self.assertEqual(model.__class__.__name__, "RandomForestClassifier")
        self.assertEqual(params["n_estimators"], 7)
        self.assertEqual(params["n_jobs"], 2)
        self.assertEqual(params["random_state"], 17)

    def test_cli_parameters_reach_decision_tree_model(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(
                    classifier="decision_tree",
                    source_profile="custom",
                    random_state=23,
                    max_depth=3,
                    decision_tree_criterion="entropy",
                    decision_tree_splitter="random",
                    decision_tree_min_samples_split=3,
                    decision_tree_min_samples_leaf=2,
                    decision_tree_max_features="sqrt",
                ),
                tmpdir,
            )

        params = result["classifier_params"]
        self.assertEqual(params["random_state"], 23)
        self.assertEqual(params["max_depth"], 3)
        self.assertEqual(params["criterion"], "entropy")
        self.assertEqual(params["splitter"], "random")
        self.assertEqual(params["min_samples_split"], 3)
        self.assertEqual(params["min_samples_leaf"], 2)
        self.assertEqual(params["max_features"], "sqrt")

    def test_cli_parameters_reach_random_forest_model(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(
                    classifier="random_forest",
                    source_profile="custom",
                    random_state=29,
                    n_estimators=5,
                    max_depth=4,
                    class_weight="balanced",
                    random_forest_criterion="entropy",
                    random_forest_min_samples_split=3,
                    random_forest_min_samples_leaf=2,
                    random_forest_max_features=None,
                    random_forest_bootstrap=False,
                    random_forest_n_jobs=1,
                ),
                tmpdir,
            )

        params = result["classifier_params"]
        self.assertEqual(params["random_state"], 29)
        self.assertEqual(params["n_estimators"], 5)
        self.assertEqual(params["max_depth"], 4)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["criterion"], "entropy")
        self.assertEqual(params["min_samples_split"], 3)
        self.assertEqual(params["min_samples_leaf"], 2)
        self.assertIsNone(params["max_features"])
        self.assertFalse(params["bootstrap"])
        self.assertEqual(params["n_jobs"], 1)

    def test_decision_tree_registers_tree_statistics(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )

        stats = result["classifier_statistics"]
        self.assertGreaterEqual(stats["max_depth"], 1)
        self.assertGreater(stats["node_count"], 0)
        self.assertGreater(stats["leaf_count"], 0)

    def test_random_forest_registers_forest_statistics(self):
        bundle = make_bundle(num_classes=3, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="random_forest", n_estimators=4, source_profile="custom"),
                tmpdir,
            )

        stats = result["classifier_statistics"]
        self.assertEqual(stats["n_estimators_trained"], 4)
        self.assertGreaterEqual(stats["mean_depth"], 0.0)
        self.assertGreaterEqual(stats["max_depth"], 0)
        self.assertGreater(stats["total_nodes"], 0)
        self.assertGreater(stats["total_leaves"], 0)

    def test_tr_tr_does_not_call_partial_fit(self):
        bundle = make_bundle(num_classes=2, rows_per_class=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )

        self.assertFalse(result["supports_partial_fit"])

    def test_unknown_model_generates_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported classifier 'unknown_model'"):
            ClassifierConfig(classifier="unknown_model")

    def test_subset_is_not_identified_as_integral_model(self):
        with self.assertRaisesRegex(ValueError, "subset classifier"):
            ClassifierConfig(classifier="decision_tree_subset")

    def test_tr_tr_execution_creates_artifact_set(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom", save_model=True),
                tmpdir,
                resolved_arguments={"classifier": "decision_tree", "random_state": numpy.int64(0)},
            )
            run_dir = Path(result["run_dir"])

            for filename in [
                "run_config.json",
                "dataset_manifest.json",
                "class_distribution_train.csv",
                "class_distribution_test.csv",
                "metrics.json",
                "per_class_metrics.csv",
                "confusion_matrix.npy",
                "resource_usage.json",
                "environment.json",
                "execution.log",
                "model.joblib",
                "status.json",
            ]:
                self.assertTrue((run_dir / filename).is_file(), filename)

            status = load_json_file(run_dir / "status.json")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["run_id"], result["run_id"])
            resource_usage = load_json_file(run_dir / "resource_usage.json")
            self.assertTrue(resource_usage["model_saved"])
            self.assertGreater(resource_usage["model_size_bytes"], 0)

    def test_tr_tr_failure_creates_failed_status(self):
        bundle = make_bundle(num_classes=2, rows_per_class=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                    "Engine.Evaluation.TrTrPipeline.build_tr_tr_classifier",
                    return_value=FailingFitClassifier()):
                result = run_tr_tr_pipeline(
                    bundle,
                    TrTrConfig(classifier="decision_tree", source_profile="custom"),
                    tmpdir,
                )

            run_dir = Path(result["run_dir"])
            status = load_json_file(run_dir / "status.json")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stage"], "training")
            self.assertEqual(status["exception_type"], "RuntimeError")
            self.assertIn("fit failed for artifact test", status["message"])

    def test_two_tr_tr_executions_do_not_overwrite_each_other(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            first = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )
            second = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )

            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertTrue(Path(first["run_dir"]).is_dir())
            self.assertTrue(Path(second["run_dir"]).is_dir())

    def test_summary_csv_receives_completed_rows(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom", random_state=11),
                tmpdir,
            )
            run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom", random_state=12),
                tmpdir,
            )

            with (Path(tmpdir) / "summary.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["classifier"], "decision_tree")
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["train_strategy"], "all")
            self.assertEqual(rows[0]["test_samples"], "8")
            self.assertIn("macro_f1", rows[0])

    def test_resolved_run_config_is_serializable(self):
        bundle = make_bundle(num_classes=2, rows_per_class=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
                resolved_arguments={
                    "np_int": numpy.int64(7),
                    "np_float": numpy.float32(0.5),
                    "path": Path(tmpdir),
                    "small_array": numpy.asarray([1, 2, 3], dtype=numpy.int64),
                },
            )
            run_config = load_json_file(Path(result["artifacts"]["run_config"]))

        self.assertEqual(run_config["resolved_arguments"]["np_int"], 7)
        self.assertEqual(run_config["resolved_arguments"]["small_array"], [1, 2, 3])

    def test_dataset_manifest_converts_numpy_types_to_json(self):
        bundle = make_bundle(num_classes=3, rows_per_class=2)
        bundle.schema.train_feature_min = numpy.float32(-0.5)
        bundle.schema.train_feature_max = numpy.float32(0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tr_tr_pipeline(
                bundle,
                TrTrConfig(classifier="decision_tree", source_profile="custom"),
                tmpdir,
            )
            manifest = load_json_file(Path(result["artifacts"]["dataset_manifest"]))

        self.assertEqual(manifest["train_feature_min"], -0.5)
        self.assertEqual(manifest["train_feature_max"], 0.5)
        self.assertEqual(manifest["splits"]["train"]["x_shape"], [6, 3])
        self.assertEqual(manifest["splits"]["test"]["class_distribution"]["2"], 2)


if __name__ == "__main__":
    unittest.main()
