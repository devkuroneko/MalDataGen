import csv
import inspect
from pathlib import Path
from unittest import mock

import numpy
import pytest

import run_appclassnet_top200 as runner
from Engine.DataIO.JsonIO import load_json_file
from Engine.DataIO.SyntheticBatchIO import SyntheticBatchWriter
from Engine.Evaluation.TrTrPipeline import TrTrConfig
from Engine.Evaluation.TrTrPipeline import run_tr_tr_pipeline
from Engine.Pipelines.Interfaces import SyntheticGenerationConfig
from Engine.Pipelines.Interfaces import build_synthetic_generation_request
from Engine.Pipelines.Interfaces import partition_generation_classes


class SpyClassifier:
    def __init__(self):
        self.fit_x = None
        self.fit_y = None

    def fit(self, x_values, y_values):
        self.fit_x = numpy.asarray(x_values)
        self.fit_y = numpy.asarray(y_values)
        return self

    def predict(self, x_values):
        return numpy.asarray(numpy.ravel(x_values[:, 0]) > 0.0, dtype=numpy.int64)

    def get_params(self, deep=True):
        return {"spy": True}


class FailingClassifier(SpyClassifier):
    def fit(self, x_values, y_values):
        raise RuntimeError("controlled failure")


def test_tr_tr_uses_real_train_real_test_and_completes(small_bundle, tmp_path):
    result = run_tr_tr_pipeline(
        small_bundle,
        TrTrConfig(classifier="decision_tree", source_profile="custom", eval_batch_size=5),
        tmp_path,
    )

    assert result["protocol"] == "TR-TR"
    assert result["source_train"] == "real"
    assert result["source_test"] == "real"
    assert result["status"] == "completed"
    assert result["status"] != "not_run", "TR-TR must never be marked not_run after successful execution"


def test_tr_tr_does_not_import_or_instantiate_generator(small_bundle, tmp_path):
    source = inspect.getsource(run_tr_tr_pipeline)
    assert "GenerativeModels" not in source

    with mock.patch.dict("sys.modules", {"Engine.Models.GenerativeModels": None}):
        result = run_tr_tr_pipeline(
            small_bundle,
            TrTrConfig(classifier="decision_tree", source_profile="custom"),
            tmp_path,
        )

    assert result["status"] == "completed"


def test_tr_tr_fit_never_sees_test_rows(small_bundle, tmp_path):
    small_bundle.test.X[:, 0] = 777.0
    spy = SpyClassifier()

    with mock.patch("Engine.Evaluation.TrTrPipeline.build_tr_tr_classifier", return_value=spy):
        run_tr_tr_pipeline(
            small_bundle,
            TrTrConfig(classifier="decision_tree", source_profile="custom"),
            tmp_path,
        )

    assert spy.fit_x is not None
    assert not numpy.any(spy.fit_x[:, 0] == 777.0), "test rows leaked into classifier.fit"


def test_tr_tr_accepts_200_classes_not_binary_only(top200_bundle, tmp_path):
    result = run_tr_tr_pipeline(
        top200_bundle,
        TrTrConfig(classifier="decision_tree", source_profile="appclassnet_top200", eval_batch_size=33),
        tmp_path,
    )

    assert result["num_classes"] == 200
    assert result["confusion_matrix_shape"] == [200, 200]


def test_tr_tr_artifacts_are_created_and_json_valid(small_bundle, tmp_path):
    result = run_tr_tr_pipeline(
        small_bundle,
        TrTrConfig(classifier="decision_tree", source_profile="custom", save_model=True),
        tmp_path,
    )
    run_dir = Path(result["run_dir"])

    for name in [
        "run_config.json",
        "dataset_manifest.json",
        "metrics.json",
        "resource_usage.json",
        "environment.json",
        "status.json",
        "model.joblib",
    ]:
        assert (run_dir / name).is_file(), f"missing artifact {name}"
    assert load_json_file(run_dir / "status.json")["status"] == "completed"
    assert result["run_id"] in run_dir.name


def test_tr_tr_failure_writes_failed_status(small_bundle, tmp_path):
    with mock.patch("Engine.Evaluation.TrTrPipeline.build_tr_tr_classifier", return_value=FailingClassifier()):
        result = run_tr_tr_pipeline(
            small_bundle,
            TrTrConfig(classifier="decision_tree", source_profile="custom"),
            tmp_path,
        )

    status = load_json_file(Path(result["run_dir"]) / "status.json")
    assert result["status"] == "failed"
    assert status["status"] == "failed"
    assert status["exception_type"] == "RuntimeError"


def test_tr_tr_summary_csv_gets_completed_row(small_bundle, tmp_path):
    result = run_tr_tr_pipeline(
        small_bundle,
        TrTrConfig(classifier="decision_tree", source_profile="custom", random_state=22),
        tmp_path,
    )

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["run_id"] == result["run_id"]
    assert rows[0]["status"] == "completed"
    assert rows[0]["classifier"] == "decision_tree"


def test_synthetic_generation_request_uses_train_valid_but_not_test(small_bundle):
    request = build_synthetic_generation_request(
        small_bundle,
        SyntheticGenerationConfig(
            generator="adversarial",
            generation_strategy="single_conditional",
            synthetic_samples_per_class=2,
            generator_transform="preserve",
        ),
    )

    assert request.train_split is small_bundle.train
    assert request.valid_split is small_bundle.valid
    assert not request.uses_test_split
    assert not hasattr(request, "test_split"), "synthetic generator request must not expose test split"


def test_synthetic_batches_preserve_features_labels_and_manifest(tmp_path):
    writer = SyntheticBatchWriter(
        root_dir=tmp_path,
        num_classes=200,
        num_features=20,
        seed=123,
        model_name="dummy",
        execution_mode="batches",
        generation_plan={"classes": {0: 2, 199: 1}, "generator_parameters": {"latent": 8}},
        requested_classes=[0, 199],
        data_space="source",
    )
    writer.write_batch(0, 0, numpy.zeros((2, 20), dtype=numpy.float32) - 0.5)
    writer.write_batch(199, 0, numpy.zeros((1, 20), dtype=numpy.float32) + 0.5)
    reader = writer.close()

    manifest = load_json_file(Path(reader.manifest_path).with_name("synthetic_manifest.json"))
    assert manifest["features"] == 20
    assert manifest["generated_per_class"] == {"0": 2, "199": 1}
    assert manifest["feature_min"] == -0.5
    assert manifest["feature_max"] == 0.5
    assert manifest["nan_count"] == 0
    assert manifest["inf_count"] == 0
    assert Path(manifest["batches"][0]["x_path"]).name == "x_00000.npy"
    assert Path(manifest["batches"][0]["y_path"]).name == "y_00000.npy"
    numpy.testing.assert_array_equal(numpy.load(manifest["batches"][1]["y_path"]), numpy.asarray([199]))


def test_synthetic_generation_groups_do_not_lose_global_classes():
    units = partition_generation_classes({0: 2, 1: 2, 198: 1, 199: 1}, "grouped_classes", 2)

    flattened = [class_id for unit in units for class_id in unit["classes"]]
    assert flattened == [0, 1, 198, 199]


def test_synthetic_samples_per_class_change_is_reflected_in_manifest(tmp_path):
    first = SyntheticBatchWriter(
        root_dir=tmp_path / "first",
        num_classes=2,
        num_features=3,
        seed=1,
        model_name="dummy",
        execution_mode="batches",
        generation_plan={"classes": {0: 1}},
    )
    first.write_batch(0, 0, numpy.ones((1, 3), dtype=numpy.float32))
    first_manifest = load_json_file(Path(first.close().manifest_path))

    second = SyntheticBatchWriter(
        root_dir=tmp_path / "second",
        num_classes=2,
        num_features=3,
        seed=1,
        model_name="dummy",
        execution_mode="batches",
        generation_plan={"classes": {0: 2}},
    )
    second.write_batch(0, 0, numpy.ones((2, 3), dtype=numpy.float32))
    second_manifest = load_json_file(Path(second.close().manifest_path))

    assert first_manifest["requested_per_class"] == {"0": 1}
    assert second_manifest["requested_per_class"] == {"0": 2}
    assert first_manifest["total_rows"] != second_manifest["total_rows"]


def resolved_cli(argv):
    args = runner.build_parser().parse_args(argv)
    runner.annotate_explicit_cli_arguments(args, argv)
    runner.apply_execution_profile(args)
    runner.apply_experiment_budget_scenario(args)
    runner.normalize_mmap_arguments(args)
    runner.normalize_classifier_arguments(args)
    runner.normalize_explicit_tr_tr_sampling(args)
    runner.validate_tr_tr_cli_arguments(args)
    return args


def test_cli_accepts_decision_tree_command():
    args = resolved_cli([
        "--pipeline", "tr_tr",
        "--classifier", "decision_tree",
        "--source_profile", "appclassnet_top200",
        "--data_format", "npy_xy",
        "--split_mode", "provided",
        "--feature_transform", "preserve",
        "--classifier_transform", "preserve",
        "--generator_transform", "preserve",
        "--evaluation_space", "source",
        "--train_sampling", "all",
        "--test_sampling", "all",
        "--dt_max_depth", "4",
    ])

    assert args.pipeline_effective == "tr_tr"
    assert args.baseline_classifier == "decision_tree"
    assert args.decision_tree_max_depth == 4


def test_cli_accepts_random_forest_command():
    args = resolved_cli([
        "--pipeline", "tr_tr",
        "--classifier", "random_forest",
        "--source_profile", "appclassnet_top200",
        "--data_format", "npy_xy",
        "--split_mode", "provided",
        "--feature_transform", "preserve",
        "--classifier_transform", "preserve",
        "--generator_transform", "preserve",
        "--evaluation_space", "source",
        "--train_sampling", "all",
        "--test_sampling", "all",
        "--rf_n_estimators", "7",
        "--rf_n_jobs", "1",
    ])

    assert args.baseline_classifier == "random_forest"
    assert args.n_estimators == 7
    assert args.random_forest_n_jobs == 1


def test_cli_tr_tr_does_not_require_synthetic_parameters():
    args = resolved_cli([
        "--pipeline", "tr_tr",
        "--classifier", "decision_tree",
        "--source_profile", "appclassnet_top200",
        "--data_format", "npy_xy",
        "--split_mode", "provided",
        "--feature_transform", "preserve",
        "--classifier_transform", "preserve",
        "--generator_transform", "preserve",
        "--evaluation_space", "source",
    ])

    assert args.train_sampling == "all"
    assert args.test_sampling == "all"
    assert args.train_samples_per_class is None
    assert args.test_samples_per_class is None


def test_cli_smoke_subset_requires_samples_per_class():
    with pytest.raises(ValueError, match="train_samples_per_class"):
        resolved_cli([
            "--pipeline", "tr_tr",
            "--classifier", "decision_tree",
            "--source_profile", "appclassnet_top200",
            "--data_format", "npy_xy",
            "--split_mode", "provided",
            "--feature_transform", "preserve",
            "--classifier_transform", "preserve",
            "--generator_transform", "preserve",
            "--evaluation_space", "source",
            "--train_sampling", "balanced_per_class",
            "--test_sampling", "all",
        ])


def test_cli_rejects_invalid_appclassnet_split_mode():
    with pytest.raises(ValueError, match="split_mode provided"):
        resolved_cli([
            "--pipeline", "tr_tr",
            "--classifier", "decision_tree",
            "--source_profile", "appclassnet_top200",
            "--data_format", "npy_xy",
            "--split_mode", "cross_validation",
            "--feature_transform", "preserve",
            "--classifier_transform", "preserve",
            "--generator_transform", "preserve",
            "--evaluation_space", "source",
        ])
