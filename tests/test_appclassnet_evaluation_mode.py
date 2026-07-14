import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import json

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData
from Engine.Evaluation.EvaluationRunner import EvaluationSplitMismatchError
from Engine.Preprocessing.FeatureTransformManager import PreprocessingSpaceMismatchError
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
from main import run_synthetic_evaluation_modes
import run_appclassnet_top200 as runner
from run_appclassnet_top200 import APPCLASSNET_NUM_CLASSES
from run_appclassnet_top200 import build_batch_main_command
from run_appclassnet_top200 import build_number_samples_per_class_plan
from run_appclassnet_top200 import deduplicate_command_options
from run_appclassnet_top200 import effective_evaluation_mode


class AppClassNetEvaluationModeTest(unittest.TestCase):

    def test_baseline_real_only_effective_evaluation_mode_is_none(self):
        arguments = SimpleNamespace(baseline_real_only=True, evaluation_mode="both")

        self.assertEqual(effective_evaluation_mode(arguments), "none")

    def test_baseline_real_only_does_not_start_synthetic_pipeline(self):
        argv = [
            "run_appclassnet_top200.py",
            "--baseline_real_only",
            "--evaluation_mode",
            "both",
            "--skip_plots",
        ]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(runner, "run_real_real_baseline", return_value=(Path("/tmp/metrics.json"), {})) as baseline, \
                mock.patch.object(runner, "write_baseline_batches_metrics") as write_metrics, \
                mock.patch.object(runner, "validate_scaler_for_campaigns", side_effect=AssertionError("synthetic validation started")), \
                mock.patch.object(runner, "preprocess_appclassnet_splits", side_effect=AssertionError("preprocessing started")), \
                mock.patch.object(runner, "run_cmd", side_effect=AssertionError("child synthetic run started")):
            exit_code = runner.main()

        self.assertEqual(exit_code, 0)
        baseline.assert_called_once()
        write_metrics.assert_called_once()

    def test_tr_ts_does_not_execute_ts_tr(self):
        owner = _FakeEvaluationOwner("tr_ts")
        synthetic = {"same": object()}

        run_synthetic_evaluation_modes(owner, {"x_evaluation_real": numpy.zeros((1, 2))}, synthetic)

        self.assertEqual(owner.calls, [("guard", synthetic), ("TR-TS", synthetic), ("skip", "TS-TR")])

    def test_ts_tr_does_not_execute_tr_ts(self):
        owner = _FakeEvaluationOwner("ts_tr")
        synthetic = {"same": object()}

        run_synthetic_evaluation_modes(owner, {"x_evaluation_real": numpy.zeros((1, 2))}, synthetic)

        self.assertEqual(owner.calls, [("guard", synthetic), ("skip", "TR-TS"), ("TS-TR", synthetic)])

    def test_both_executes_both_with_split_synthetic_objects(self):
        owner = _FakeEvaluationOwner("both")
        synthetic = SimpleNamespace(train_reader={"split": "train"}, test_reader={"split": "test"})

        run_synthetic_evaluation_modes(owner, {"x_evaluation_real": numpy.zeros((1, 2))}, synthetic)

        self.assertEqual(
            owner.calls,
            [
                ("guard", synthetic.test_reader),
                ("guard", synthetic.train_reader),
                ("TR-TS", synthetic.test_reader),
                ("TS-TR", synthetic.train_reader),
            ],
        )

    def test_provided_tr_ts_uses_train_and_synthetic_test(self):
        bundle = _provided_bundle()
        owner = _FakeEvaluationOwner("tr_ts", split_mode="provided")
        owner._dataset_bundle = bundle
        synthetic = SimpleNamespace(train_reader={"split": "synthetic_train"}, test_reader={"split": "synthetic_test"})

        run_synthetic_evaluation_modes(owner, {"dataset_bundle": bundle}, synthetic)

        self.assertEqual(owner.calls[-2:], [("TR-TS", "train", synthetic.test_reader), ("skip", "TS-TR")])

    def test_provided_ts_tr_uses_synthetic_train_and_test(self):
        bundle = _provided_bundle()
        owner = _FakeEvaluationOwner("ts_tr", split_mode="provided")
        owner._dataset_bundle = bundle
        synthetic = SimpleNamespace(train_reader={"split": "synthetic_train"}, test_reader={"split": "synthetic_test"})

        run_synthetic_evaluation_modes(owner, {"dataset_bundle": bundle}, synthetic)

        self.assertEqual(owner.calls[-2:], [("skip", "TR-TS"), ("TS-TR", synthetic.train_reader, "test")])

    def test_provided_both_never_routes_valid_to_ts_tr(self):
        bundle = _provided_bundle()
        owner = _FakeEvaluationOwner("both", split_mode="provided")
        owner._dataset_bundle = bundle
        synthetic = SimpleNamespace(train_reader={"split": "synthetic_train"}, test_reader={"split": "synthetic_test"})

        run_synthetic_evaluation_modes(owner, {"dataset_bundle": bundle}, synthetic)

        self.assertIn(("TS-TR", synthetic.train_reader, "test"), owner.calls)
        self.assertNotIn(("TS-TR", synthetic.train_reader, "valid"), owner.calls)

    def test_ts_tr_rejects_valid_as_real_test_split(self):
        owner = _FakeEvaluationOwner("ts_tr", split_mode="provided")

        with self.assertRaisesRegex(EvaluationSplitMismatchError, "TS-TR requires real split 'test', but received 'valid'"):
            owner.evaluation_TS_TR(
                synthetic_train_data={"split": "synthetic_train"},
                real_test_data=_provided_bundle().valid,
            )

    def test_scale_mismatch_blocks_evaluation(self):
        real = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        synthetic = numpy.array([[0.0, 1.0], [0.5, 0.75]], dtype=numpy.float32)

        with self.assertRaises(PreprocessingSpaceMismatchError):
            ScaleGuard.validate_before_evaluation(
                real,
                synthetic,
                ScaleGuard.describe(real, data_space="source"),
                ScaleGuard.describe(synthetic, data_space="generator", transform_id="abc"),
                context="TR-TS",
            )

    def test_batch_command_propagates_evaluation_mode_and_synthetic_quotas(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            for split in ("train", "valid", "test"):
                numpy.save(raw_root / f"{split}_x.npy", numpy.zeros((APPCLASSNET_NUM_CLASSES, 2), dtype=numpy.float32))
                numpy.save(raw_root / f"{split}_y.npy", numpy.arange(APPCLASSNET_NUM_CLASSES, dtype=numpy.int64))

            args = _batch_args(evaluation_mode="tr_ts")
            args.synthetic_train_samples_per_class = 17
            args.synthetic_test_samples_per_class = 11
            command = build_batch_main_command(
                "python3",
                raw_root,
                Path(directory) / "out",
                {"model_type": "copy"},
                20,
                "multiclass",
                args,
            )

        self.assertIn("--evaluation_mode", command)
        self.assertEqual(command[command.index("--evaluation_mode") + 1], "tr_ts")
        self.assertEqual(
            command[command.index("--train_samples_per_class") + 1],
            str(runner.DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS),
        )
        self.assertEqual(
            command[command.index("--test_samples_per_class") + 1],
            str(runner.DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS),
        )
        self.assertIn("--synthetic_train_samples_per_class", command)
        self.assertIn("--synthetic_test_samples_per_class", command)
        self.assertEqual(command[command.index("--number_samples_per_class") + 1], build_number_samples_per_class_plan(28))

    def test_no_sample_arguments_use_defaults(self):
        command = _batch_command_for_args(self, _batch_args(), {"model_type": "copy"})

        self.assertEqual(
            command[command.index("--train_samples_per_class") + 1],
            str(runner.DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS),
        )
        self.assertEqual(
            command[command.index("--test_samples_per_class") + 1],
            str(runner.DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS),
        )
        self.assertEqual(
            command[command.index("--synthetic_train_samples_per_class") + 1],
            str(runner.DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS),
        )
        self.assertEqual(
            command[command.index("--synthetic_test_samples_per_class") + 1],
            str(runner.DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS),
        )
        self.assertEqual(
            command[command.index("--number_samples_per_class") + 1],
            build_number_samples_per_class_plan(runner.DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS * 2),
        )

    def test_campaign_sample_arguments_override_defaults(self):
        command = _batch_command_for_args(
            self,
            _batch_args(),
            {
                "model_type": "copy",
                "train_samples_per_class": 80,
                "test_samples_per_class": 40,
                "synthetic_train_samples_per_class": 12,
                "synthetic_test_samples_per_class": 13,
                "generated_samples_per_class": 30,
            },
        )

        self.assertEqual(command[command.index("--train_samples_per_class") + 1], "80")
        self.assertEqual(command[command.index("--test_samples_per_class") + 1], "40")
        self.assertEqual(command[command.index("--synthetic_train_samples_per_class") + 1], "12")
        self.assertEqual(command[command.index("--synthetic_test_samples_per_class") + 1], "13")
        self.assertEqual(command[command.index("--number_samples_per_class") + 1], build_number_samples_per_class_plan(30))

    def test_cli_sample_arguments_override_campaign(self):
        args = _batch_args()
        args.train_samples_per_class = 1000
        args.synthetic_train_samples_per_class = 500
        args.synthetic_test_samples_per_class = 500
        args.generated_samples_per_class = 1000

        command = _batch_command_for_args(
            self,
            args,
            {
                "model_type": "copy",
                "train_samples_per_class": 500,
                "synthetic_train_samples_per_class": 100,
                "synthetic_test_samples_per_class": 100,
                "generated_samples_per_class": 200,
            },
        )

        self.assertEqual(command[command.index("--train_samples_per_class") + 1], "1000")
        self.assertEqual(command[command.index("--synthetic_train_samples_per_class") + 1], "500")
        self.assertEqual(command[command.index("--synthetic_test_samples_per_class") + 1], "500")
        self.assertEqual(command[command.index("--number_samples_per_class") + 1], build_number_samples_per_class_plan(1000))

    def test_synthetic_50_50_plans_100_per_class(self):
        self.assertEqual(_planned_samples_for_synthetic_quotas(self, 50, 50), build_number_samples_per_class_plan(100))

    def test_synthetic_100_100_plans_200_per_class(self):
        self.assertEqual(_planned_samples_for_synthetic_quotas(self, 100, 100), build_number_samples_per_class_plan(200))

    def test_synthetic_200_200_plans_400_per_class(self):
        self.assertEqual(_planned_samples_for_synthetic_quotas(self, 200, 200), build_number_samples_per_class_plan(400))

    def test_synthetic_500_500_plans_1000_per_class(self):
        self.assertEqual(_planned_samples_for_synthetic_quotas(self, 500, 500), build_number_samples_per_class_plan(1000))

    def test_batch_command_builds_for_supported_synthetic_sizes(self):
        for samples_per_split in (50, 100, 200, 256, 500):
            with self.subTest(samples_per_split=samples_per_split):
                plan = _planned_samples_for_synthetic_quotas(self, samples_per_split, samples_per_split)
                self.assertEqual(plan, build_number_samples_per_class_plan(samples_per_split * 2))

    def test_train_samples_per_class_1000_reaches_main_as_1000(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            for split in ("train", "valid", "test"):
                numpy.save(raw_root / f"{split}_x.npy", numpy.zeros((APPCLASSNET_NUM_CLASSES, 2), dtype=numpy.float32))
                numpy.save(raw_root / f"{split}_y.npy", numpy.arange(APPCLASSNET_NUM_CLASSES, dtype=numpy.int64))

            args = _batch_args()
            args.train_samples_per_class = 1000
            args.synthetic_train_samples_per_class = 500
            args.synthetic_test_samples_per_class = 500
            args._explicit_cli_options = {"train_samples_per_class", "synthetic_train_samples_per_class", "synthetic_test_samples_per_class"}
            command = build_batch_main_command(
                "python3",
                raw_root,
                Path(directory) / "out",
                {"model_type": "copy", "train_samples_per_class": 500},
                20,
                "multiclass",
                args,
            )

        self.assertEqual(command[command.index("--train_samples_per_class") + 1], "1000")
        self.assertEqual(command[command.index("--synthetic_train_samples_per_class") + 1], "500")

    def test_generated_samples_per_class_sufficient_is_used_for_legacy_plan(self):
        args = _batch_args()
        args.synthetic_train_samples_per_class = 100
        args.synthetic_test_samples_per_class = 100
        args.generated_samples_per_class = 250

        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        self.assertEqual(command[command.index("--number_samples_per_class") + 1], build_number_samples_per_class_plan(250))

    def test_generated_samples_per_class_insufficient_fails_before_subprocess(self):
        args = _batch_args()
        args.synthetic_train_samples_per_class = 200
        args.synthetic_test_samples_per_class = 200
        args.generated_samples_per_class = 399

        with self.assertRaisesRegex(ValueError, "InsufficientGeneratedSamplesPerClass"):
            _batch_command_for_args(self, args, {"model_type": "copy"})

    def test_duplicate_arguments_are_detected(self):
        with self.assertRaisesRegex(ValueError, "DuplicateCommandArgumentConflict"):
            deduplicate_command_options([
                "python3",
                "main.py",
                "--variational_autoencoder_batch_size",
                "128",
                "--variational_autoencoder_batch_size",
                "8192",
            ])

    def test_builtin_batch_override_removes_duplicate_model_batch_size(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            for split in ("train", "valid", "test"):
                numpy.save(raw_root / f"{split}_x.npy", numpy.zeros((APPCLASSNET_NUM_CLASSES, 2), dtype=numpy.float32))
                numpy.save(raw_root / f"{split}_y.npy", numpy.arange(APPCLASSNET_NUM_CLASSES, dtype=numpy.int64))

            args = _batch_args()
            args.batch_size = 8192
            command = build_batch_main_command(
                "python3",
                raw_root,
                Path(directory) / "out",
                {"model_type": "variational", "variational_autoencoder_batch_size": 128},
                20,
                "multiclass",
                args,
            )

        self.assertEqual(command.count("--variational_autoencoder_batch_size"), 1)
        self.assertEqual(command[command.index("--variational_autoencoder_batch_size") + 1], "8192")

    def test_batch_command_contains_no_duplicate_arguments(self):
        command = _batch_command_for_args(
            self,
            _batch_args(),
            {"model_type": "variational", "variational_autoencoder_batch_size": 128},
        )
        options = [token for token in command if str(token).startswith("--")]

        self.assertEqual(len(options), len(set(options)))

    def test_legacy_number_samples_per_class_plan_uses_generated_count(self):
        plan = build_number_samples_per_class_plan(123)

        self.assertTrue(plan.startswith("0:123,1:123,2:123"))
        self.assertTrue(plan.endswith("197:123,198:123,199:123"))
        self.assertEqual(len(plan.split(",")), APPCLASSNET_NUM_CLASSES)

    def test_runner_parser_sample_defaults_are_none(self):
        parsed = runner.build_parser().parse_args([])

        self.assertIsNone(parsed.train_samples_per_class)
        self.assertIsNone(parsed.test_samples_per_class)
        self.assertIsNone(parsed.synthetic_train_samples_per_class)
        self.assertIsNone(parsed.synthetic_test_samples_per_class)
        self.assertIsNone(parsed.generated_samples_per_class)

    def test_evaluation_mode_none(self):
        _, payload = runner.write_batches_metrics(
            [],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="none"),
        )

        self.assertEqual(payload["TR-TR"]["status"], "not_run")
        self.assertEqual(payload["TR-TS"]["status"], "not_run")
        self.assertEqual(payload["TS-TR"]["status"], "not_run")
        self.assertIsNone(payload["TR-TS"]["Accuracy"])
        self.assertIsNone(payload["TS-TR"]["Accuracy"])

    def test_evaluation_mode_tr_ts(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=True, include_ts_tr=False)

        _, payload = runner.write_batches_metrics(
            [results_path],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="tr_ts"),
        )

        self.assertEqual(payload["TR-TS"]["status"], "completed")
        self.assertEqual(payload["TS-TR"]["status"], "not_run")
        self.assertIsNotNone(payload["TR-TS"]["Accuracy"])

    def test_evaluation_mode_ts_tr(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=False, include_ts_tr=True)

        _, payload = runner.write_batches_metrics(
            [results_path],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="ts_tr"),
        )

        self.assertEqual(payload["TR-TS"]["status"], "not_run")
        self.assertEqual(payload["TS-TR"]["status"], "completed")
        self.assertIsNotNone(payload["TS-TR"]["Accuracy"])

    def test_evaluation_mode_both(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=True, include_ts_tr=True)

        _, payload = runner.write_batches_metrics(
            [results_path],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="both"),
        )

        self.assertEqual(payload["TR-TR"]["status"], "not_run")
        self.assertEqual(payload["TR-TS"]["status"], "completed")
        self.assertEqual(payload["TS-TR"]["status"], "completed")
        self.assertIsNotNone(payload["TR-TS"]["MacroF1"])
        self.assertIsNotNone(payload["TS-TR"]["WeightedF1"])

    def test_baseline_real_only(self):
        _, payload = runner.write_baseline_batches_metrics(
            {
                "classifier": "DecisionTree",
                "classifier_params": {"max_depth": 3},
                "train_class_counts": {"0": 2, "1": 2, "2": 2},
                "test_class_counts": {"0": 1, "1": 1, "2": 1},
                "data_space": "source",
                "feature_range": {"train": [0, 1], "test": [0, 1]},
                "duration_seconds": 0.01,
                "metrics": {
                    "Accuracy": 1.0,
                    "BalancedAccuracy": 1.0,
                    "MacroF1": 1.0,
                    "WeightedF1": 1.0,
                },
            },
            _temp_metrics_path(self),
        )

        self.assertEqual(payload["TR-TR"]["status"], "completed")
        self.assertEqual(payload["TR-TS"]["status"], "not_run")
        self.assertEqual(payload["TS-TR"]["status"], "not_run")

    def test_requested_evaluation_cannot_finish_not_run(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=False, include_ts_tr=False)

        with self.assertRaisesRegex(RuntimeError, "Requested evaluation TR-TS was not completed"):
            runner.write_batches_metrics(
                [results_path],
                _temp_metrics_path(self),
                SimpleNamespace(evaluation_mode="tr_ts"),
            )

    def test_result_is_serialized_once(self):
        with mock.patch("builtins.print") as print_mock:
            runner.print_results_payload({"ok": True})

        print_mock.assert_called_once()

    def test_both_uses_separate_synthetic_artifacts(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=True, include_ts_tr=True)

        _, payload = runner.write_batches_metrics(
            [results_path],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="both"),
        )

        self.assertIn("/test/manifest.json", payload["TR-TS"]["synthetic_manifest"])
        self.assertIn("/train/manifest.json", payload["TS-TR"]["synthetic_manifest"])

    def test_timestamped_run_dir_is_reused_for_results(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "dataset" / "campaign" / "combination_1"
            actual = Path(f"{base}_2026-07-13_23-44-10")
            results_dir = actual / "EvaluationResults"
            results_dir.mkdir(parents=True)
            (results_dir / "Results.json").write_text("{}", encoding="utf-8")

            self.assertEqual(runner.resolve_actual_run_dir(base), actual)

    def test_run_artifacts_keeps_real_results_json_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "dataset" / "campaign" / "combination_1"
            actual = Path(f"{base}_2026-07-13_23-44-10")
            (actual / "EvaluationResults").mkdir(parents=True)
            (actual / "EvaluationResults" / "Results.json").write_text("{}", encoding="utf-8")
            (actual / "DataGenerated" / "synthetic_batches" / "train").mkdir(parents=True)
            (actual / "DataGenerated" / "synthetic_batches" / "test").mkdir(parents=True)
            _write_manifest(actual / "DataGenerated" / "synthetic_batches" / "train" / "manifest.json", "train")
            _write_manifest(actual / "DataGenerated" / "synthetic_batches" / "test" / "manifest.json", "test")

            artifacts = runner.build_run_artifacts(base, {"model_type": "copy"})

        self.assertEqual(artifacts.combination_dir, actual)
        self.assertEqual(artifacts.results_json_path, actual / "EvaluationResults" / "Results.json")
        self.assertEqual(artifacts.synthetic_train_manifest_path.name, "manifest.json")
        self.assertEqual(artifacts.synthetic_test_manifest_path.name, "manifest.json")

    def test_missing_synthetic_train_fails_ts_tr(self):
        synthetic = _SyntheticMemoryBatches({0: numpy.zeros((1, 2), dtype=numpy.float32)})

        with self.assertRaisesRegex(ValueError, "TS-TR synthetic data"):
            from Engine.Classifiers.BatchClassifiers import validate_synthetic_batches_for_evaluation

            validate_synthetic_batches_for_evaluation(
                synthetic,
                "TS-TR",
                expected_num_classes=3,
                samples_per_class=2,
                expected_num_features=2,
            )

    def test_missing_synthetic_test_fails_tr_ts(self):
        synthetic = _SyntheticMemoryBatches({0: numpy.zeros((1, 2), dtype=numpy.float32)})

        with self.assertRaisesRegex(ValueError, "TR-TS synthetic data"):
            from Engine.Classifiers.BatchClassifiers import validate_synthetic_batches_for_evaluation

            validate_synthetic_batches_for_evaluation(
                synthetic,
                "TR-TS",
                expected_num_classes=3,
                samples_per_class=2,
                expected_num_features=2,
            )


class _FakeEvaluationOwner:
    def __init__(self, evaluation_mode, split_mode="cross_validation"):
        self.arguments = SimpleNamespace(evaluation_mode=evaluation_mode, split_mode=split_mode)
        self.fold_number = 0
        self.calls = []

    def _guard_current_evaluation_space(self, dictionary_data, synthetic_data):
        self.calls.append(("guard", synthetic_data))

    def evaluation_TR_TS(self, dictionary_data=None, synthetic_data=None, *, real_train_data=None, synthetic_test_data=None):
        if real_train_data is not None:
            self.calls.append(("TR-TS", real_train_data.name, synthetic_test_data))
        else:
            self.calls.append(("TR-TS", synthetic_data))

    def evaluation_TS_TR(self, dictionary_data=None, synthetic_data=None, *, synthetic_train_data=None, real_test_data=None):
        if real_test_data is not None:
            if real_test_data.name != "test":
                raise EvaluationSplitMismatchError(
                    f"TS-TR requires real split 'test', but received '{real_test_data.name}'."
                )
            self.calls.append(("TS-TR", synthetic_train_data, real_test_data.name))
        else:
            self.calls.append(("TS-TR", synthetic_data))

    def mark_evaluation_classifiers_not_applicable(self, evaluation_type, fold, reason):
        self.calls.append(("skip", evaluation_type))


def _provided_bundle():
    schema = DatasetSchema(
        feature_names=["f0", "f1"],
        target_type="multiclass",
        feature_type="continuous",
        num_classes=3,
        source_format="npy_xy",
        source_profile="appclassnet_top200",
    )
    return DatasetBundle(
        train=SplitData(
            X=numpy.zeros((6, 2), dtype=numpy.float32),
            y=numpy.array([0, 0, 1, 1, 2, 2]),
            name="train",
            x_path="/dataset/train_x.npy",
            y_path="/dataset/train_y.npy",
            dataset_id="appclassnet_top200",
        ),
        valid=SplitData(
            X=numpy.ones((3, 2), dtype=numpy.float32),
            y=numpy.array([0, 1, 2]),
            name="valid",
            x_path="/dataset/valid_x.npy",
            y_path="/dataset/valid_y.npy",
            dataset_id="appclassnet_top200",
        ),
        test=SplitData(
            X=numpy.full((9, 2), 2.0, dtype=numpy.float32),
            y=numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            name="test",
            x_path="/dataset/test_x.npy",
            y_path="/dataset/test_y.npy",
            dataset_id="appclassnet_top200",
        ),
        schema=schema,
    )


class _SyntheticMemoryBatches:
    def __init__(self, batches_by_class):
        self.batches_by_class = batches_by_class

    def iter_batches(self):
        for label, values in self.batches_by_class.items():
            yield label, values


def _batch_args(evaluation_mode="both"):
    return SimpleNamespace(
        dataset_split="train",
        batch_size=64,
        eval_batch_size=64,
        generation_batch_size=64,
        eval_classifier="decision_tree_subset",
        batch_classifier_subset_size=100,
        min_samples_per_class_required=1,
        generation_strategy="single_conditional",
        classes_per_group=10,
        source_profile="appclassnet_top200",
        generator_transform="preserve",
        classifier_transform="preserve",
        evaluation_space="source",
        evaluation_mode=evaluation_mode,
        use_mmap=True,
        max_train_samples=None,
        max_samples_per_class=None,
        train_samples_per_class=None,
        test_samples_per_class=None,
        synthetic_train_samples_per_class=None,
        synthetic_test_samples_per_class=None,
        generated_samples_per_class=None,
        n_estimators=None,
        max_depth=None,
        max_samples=None,
        class_weight=None,
        full=False,
        dry_run_memory=False,
        save_synthetic_format=None,
        materialize_synthetic=False,
        allow_double_transform=False,
        allow_scaler_refit=False,
        inverse_transform_synthetic=True,
    )


def _batch_command_for_args(test_case, args, combination):
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    raw_root = Path(directory.name)
    for split in ("train", "valid", "test"):
        numpy.save(raw_root / f"{split}_x.npy", numpy.zeros((APPCLASSNET_NUM_CLASSES, 2), dtype=numpy.float32))
        numpy.save(raw_root / f"{split}_y.npy", numpy.arange(APPCLASSNET_NUM_CLASSES, dtype=numpy.int64))

    return build_batch_main_command(
        "python3",
        raw_root,
        raw_root / "out",
        combination,
        20,
        "multiclass",
        args,
    )


def _planned_samples_for_synthetic_quotas(test_case, train_quota, test_quota):
    args = _batch_args()
    args.synthetic_train_samples_per_class = train_quota
    args.synthetic_test_samples_per_class = test_quota
    command = _batch_command_for_args(test_case, args, {"model_type": "copy"})
    return command[command.index("--number_samples_per_class") + 1]


def _temp_metrics_path(test_case):
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    return Path(directory.name) / "metrics.json"


def _write_batch_results_fixture(test_case, include_tr_ts=True, include_ts_tr=True):
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    run_dir = Path(directory.name) / "dataset" / "campaign" / "combination_1"
    results_dir = run_dir / "EvaluationResults"
    results_dir.mkdir(parents=True)
    for split in ("train", "test"):
        _write_manifest(run_dir / "DataGenerated" / "synthetic_batches" / split / "manifest.json", split)

    results = {"BatchClassifier": {"1-Fold": {}}}
    if include_tr_ts:
        results["TR-TS"] = {"DecisionTreeSubset": {"1-Fold": _metric_block()}}
        results["BatchClassifier"]["1-Fold"]["TR-TS"] = _metadata_block()
    if include_ts_tr:
        results["TS-TR"] = {"DecisionTreeSubset": {"1-Fold": _metric_block()}}
        results["BatchClassifier"]["1-Fold"]["TS-TR"] = _metadata_block()
    results_path = results_dir / "Results.json"
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file)
    return results_path


def _write_manifest(manifest_path, split):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    batches_by_class = {}
    for class_id in range(3):
        class_dir = manifest_path.parent / f"class_{class_id:03d}"
        class_dir.mkdir(exist_ok=True)
        batch_path = class_dir / "batch_000000.npy"
        numpy.save(batch_path, numpy.full((2, 2), class_id, dtype=numpy.float32))
        batches_by_class[str(class_id)] = [{"path": str(batch_path), "shape": [2, 2]}]
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "total_rows": 6,
                "num_classes": 3,
                "features": 2,
                "batches_by_class": batches_by_class,
                "data_space": "source",
                "transform_id": None,
                "transform_history": [],
                "split": split,
                "seed": 42 if split == "train" else 43,
                "model": "copy",
                "fold": 1,
            },
            manifest_file,
        )


def _metric_block():
    return {
        "Accuracy": 1.0,
        "BalancedAccuracy": 1.0,
        "MacroF1": 1.0,
        "WeightedF1": 1.0,
    }


def _metadata_block():
    return {
        "classifier": "DecisionTreeSubset",
        "n_estimators": None,
        "max_depth": None,
        "max_samples": None,
        "class_weight": None,
        "subset_quota_per_class": 2,
        "real_samples_used_by_class": {"0": 2, "1": 2, "2": 2},
        "synthetic_samples_used_by_class": {"0": 2, "1": 2, "2": 2},
        "training_time_seconds": 0.01,
        "evaluation_time_seconds": 0.02,
    }


if __name__ == "__main__":
    unittest.main()
