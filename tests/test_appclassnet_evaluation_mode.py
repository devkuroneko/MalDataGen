import tempfile
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import json

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData
from Engine.Classifiers.BatchClassifiers import train_batch_classifier
from Engine.Evaluation.EvaluationRunner import EvaluationSplitMismatchError
from Engine.Preprocessing.FeatureTransformManager import PreprocessingSpaceMismatchError
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
from main import run_synthetic_evaluation_modes
import run_appclassnet_top200 as runner
from run_appclassnet_top200 import APPCLASSNET_NUM_CLASSES
from run_appclassnet_top200 import build_batch_main_command
from run_appclassnet_top200 import build_main_command
from run_appclassnet_top200 import build_number_samples_per_class_plan
from run_appclassnet_top200 import deduplicate_command_options
from run_appclassnet_top200 import effective_evaluation_mode


class AppClassNetEvaluationModeTest(unittest.TestCase):

    def test_no_arguments_resolve_to_full(self):
        args = _parsed_runner_args([])

        self.assertEqual(args.run_mode_effective, "full")
        self.assertEqual(args.run_mode_origin, "default")
        self.assertEqual(args.pipeline_effective, "all")

    def test_campaign_sf_resolves_to_demo(self):
        args = _parsed_runner_args(["-c", "sf"])

        self.assertEqual(args.run_mode_effective, "demo")
        self.assertEqual(args.run_mode_origin, "legacy_campaign")

    def test_explicit_run_mode_demo_resolves_to_demo(self):
        args = _parsed_runner_args(["--run_mode", "demo"])

        self.assertEqual(args.run_mode_effective, "demo")

    def test_explicit_run_mode_full_resolves_to_full(self):
        args = _parsed_runner_args(["--run_mode", "full"])

        self.assertEqual(args.run_mode_effective, "full")

    def test_explicit_run_mode_wins_over_campaign(self):
        args = _parsed_runner_args(["--run_mode", "full", "-c", "sf"])

        self.assertEqual(args.run_mode_effective, "full")
        self.assertEqual(runner.choose_campaigns(args.campaign, full=args.run_mode_effective == "full"), runner.DEFAULT_CAMPAIGN)

    def test_legacy_full_resolves_to_full(self):
        with self.assertWarns(DeprecationWarning):
            args = _parsed_runner_args(["--full", "-c", "sf"])

        self.assertEqual(args.run_mode_effective, "full")
        self.assertEqual(args.run_mode_origin, "legacy_full")

    def test_demo_does_not_mutate_full_defaults(self):
        demo = _parsed_runner_args(["-c", "sf"])
        full = _parsed_runner_args([])

        self.assertEqual(demo.execution_mode, "batches")
        self.assertEqual(full.execution_mode, "normal")

    def test_full_does_not_inherit_demo_limits(self):
        args = _parsed_runner_args([])

        self.assertEqual(runner._profile_parameter_value(args, "train_samples_per_class"), 100000)
        self.assertNotEqual(runner._profile_parameter_value(args, "train_samples_per_class"), 1000)

    def test_pipeline_tr_tr_executes_only_tr_tr(self):
        args = _parsed_runner_args(["--run_mode", "full", "--pipeline", "tr_tr"])
        plan = runner.build_pipeline_plan(args.pipeline_effective)

        self.assertTrue(args.baseline_real_only)
        self.assertEqual(args.evaluation_mode, "none")
        self.assertEqual(plan["run_tr_tr"], "completed")
        self.assertEqual(plan["run_tr_ts"], "not_run")
        self.assertEqual(plan["run_ts_tr"], "not_run")

    def test_parser_accepts_explicit_tr_tr_decision_tree_command(self):
        args = _resolved_tr_tr_args([
            "--pipeline", "tr_tr",
            "--source_profile", "appclassnet_top200",
            "--data_format", "npy_xy",
            "--split_mode", "provided",
            "--execution_mode", "batches",
            "--mmap_npy",
            "--feature_transform", "preserve",
            "--classifier_transform", "preserve",
            "--evaluation_space", "source",
            "--classifier", "decision_tree",
            "--train_sampling", "all",
            "--test_sampling", "all",
            "--eval_batch_size", "32",
            "--random_state", "7",
            "--dt_criterion", "entropy",
            "--dt_splitter", "random",
            "--dt_max_depth", "4",
            "--dt_min_samples_split", "3",
            "--dt_min_samples_leaf", "2",
            "--dt_max_features", "sqrt",
            "--dt_class_weight", "balanced",
        ])

        self.assertEqual(args.baseline_classifier, "decision_tree")
        self.assertTrue(args.use_mmap)
        self.assertEqual(args.train_sampling, "all")
        self.assertEqual(args.test_sampling, "all")
        self.assertEqual(args.decision_tree_criterion, "entropy")
        self.assertEqual(args.decision_tree_max_depth, 4)

    def test_parser_accepts_explicit_tr_tr_random_forest_command(self):
        args = _resolved_tr_tr_args([
            "--pipeline", "tr_tr",
            "--execution_mode", "batches",
            "--use_mmap",
            "--classifier", "random_forest",
            "--train_sampling", "all",
            "--test_sampling", "all",
            "--rf_n_estimators", "11",
            "--rf_criterion", "entropy",
            "--rf_max_depth", "6",
            "--rf_min_samples_split", "3",
            "--rf_min_samples_leaf", "2",
            "--rf_max_features", "log2",
            "--no-rf_bootstrap",
            "--rf_class_weight", "balanced_subsample",
            "--rf_n_jobs", "2",
        ])

        self.assertEqual(args.baseline_classifier, "random_forest")
        self.assertEqual(args.n_estimators, 11)
        self.assertEqual(args.random_forest_criterion, "entropy")
        self.assertEqual(args.random_forest_max_depth, 6)
        self.assertFalse(args.random_forest_bootstrap)
        self.assertEqual(args.random_forest_class_weight, "balanced_subsample")
        self.assertEqual(args.random_forest_n_jobs, 2)

    def test_explicit_tr_tr_preserve_is_resolved_without_synthetic_arguments(self):
        args = _resolved_tr_tr_args([
            "--pipeline", "tr_tr",
            "--execution_mode", "batches",
            "--mmap_npy",
            "--classifier", "decision_tree",
        ])

        self.assertEqual(args.feature_transform, "preserve")
        self.assertEqual(args.generator_transform, "preserve")
        self.assertEqual(args.classifier_transform, "preserve")
        self.assertEqual(args.evaluation_space, "source")
        self.assertEqual(args.train_sampling, "all")
        self.assertEqual(args.test_sampling, "all")
        self.assertIsNone(args.train_samples_per_class)
        self.assertIsNone(args.test_samples_per_class)

    def test_random_forest_cli_parameters_reach_model(self):
        args = _resolved_tr_tr_args([
            "--pipeline", "tr_tr",
            "--classifier", "random_forest",
            "--rf_n_estimators", "5",
            "--rf_max_depth", "3",
            "--rf_n_jobs", "1",
            "--random_state", "19",
        ])
        from Engine.Evaluation.TrTrPipeline import tr_tr_config_from_namespace

        model = runner.build_tr_tr_classifier(tr_tr_config_from_namespace(args))
        params = model.get_params(deep=True)

        self.assertEqual(params["n_estimators"], 5)
        self.assertEqual(params["max_depth"], 3)
        self.assertEqual(params["n_jobs"], 1)
        self.assertEqual(params["random_state"], 19)

    def test_invalid_tr_tr_configuration_fails_before_dataset_loading(self):
        args = _resolved_tr_tr_args([
            "--pipeline", "tr_tr",
            "--classifier", "decision_tree",
        ], validate=False)
        args.eval_batch_size = 0

        with mock.patch.object(runner, "validate_raw_split", side_effect=AssertionError("dataset should not load")):
            with self.assertRaisesRegex(ValueError, "eval_batch_size"):
                runner.validate_tr_tr_cli_arguments(args)

    def test_old_csv_style_command_is_still_accepted(self):
        args = _parsed_runner_args([
            "--run_mode", "demo",
            "--pipeline", "synthetic",
            "--execution_mode", "normal",
            "--data_format", "csv",
            "--split_mode", "cross_validation",
        ])

        self.assertEqual(args.pipeline_effective, "synthetic")
        self.assertEqual(args.execution_mode, "normal")
        self.assertEqual(args.data_format, "csv")
        self.assertEqual(args.split_mode, "cross_validation")

    def test_pipeline_synthetic_executes_tr_ts_and_ts_tr(self):
        args = _parsed_runner_args(["--run_mode", "demo", "--pipeline", "synthetic"])
        plan = runner.build_pipeline_plan(args.pipeline_effective)

        self.assertFalse(args.run_tr_tr_effective)
        self.assertEqual(args.evaluation_mode, "both")
        self.assertEqual(plan["run_tr_ts"], "completed")
        self.assertEqual(plan["run_ts_tr"], "completed")
        self.assertEqual(plan["run_tr_ts_tr"], "not_run")

    def test_pipeline_augmentation_executes_only_tr_ts_tr(self):
        args = _parsed_runner_args(["--run_mode", "demo", "--pipeline", "augmentation"])
        plan = runner.build_pipeline_plan(args.pipeline_effective)

        self.assertFalse(args.run_tr_tr_effective)
        self.assertEqual(args.evaluation_mode, "tr_ts_tr")
        self.assertEqual(plan["run_tr_ts"], "not_run")
        self.assertEqual(plan["run_ts_tr"], "not_run")
        self.assertEqual(plan["run_tr_ts_tr"], "completed")

    def test_pipeline_augmentation_does_not_inherit_synthetic_test_quota(self):
        args = _parsed_runner_args(["--run_mode", "demo", "--pipeline", "augmentation"])
        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        self.assertEqual(command[command.index("--synthetic_train_samples_per_class") + 1], "50")
        self.assertEqual(command[command.index("--synthetic_test_samples_per_class") + 1], "0")
        self.assertEqual(command[command.index("--number_samples_per_class") + 1], build_number_samples_per_class_plan(50))

    def test_pipeline_all_executes_all_evaluations(self):
        args = _parsed_runner_args(["--run_mode", "full", "--pipeline", "all"])
        plan = runner.build_pipeline_plan(args.pipeline_effective)

        self.assertTrue(args.run_tr_tr_effective)
        self.assertEqual(args.evaluation_mode, "all")
        self.assertEqual(plan["run_tr_tr"], "completed")
        self.assertEqual(plan["run_tr_ts"], "completed")
        self.assertEqual(plan["run_ts_tr"], "completed")
        self.assertEqual(plan["run_tr_ts_tr"], "completed")

    def test_pipeline_all_reuses_preprocessing(self):
        plan = runner.build_pipeline_plan("all")

        self.assertEqual(plan["preprocessing_runs"], 1)

    def test_pipeline_all_reuses_synthetic_generation(self):
        plan = runner.build_pipeline_plan("all")

        self.assertEqual(plan["synthetic_generation_runs"], 1)

    def test_demo_output_directory_contains_demo(self):
        args = _parsed_runner_args(["-c", "sf"])

        self.assertIn("/demo/", str(runner.build_output_directory(args)))

    def test_full_output_directory_contains_full(self):
        args = _parsed_runner_args([])

        self.assertIn("/full/", str(runner.build_output_directory(args)))

    def test_run_results_contains_mode(self):
        args = _parsed_runner_args(["--run_mode", "demo", "--pipeline", "synthetic"])
        with tempfile.TemporaryDirectory() as directory:
            _, payload = runner.write_run_results(
                Path(directory),
                args,
                ["variational_demo"],
                {
                    "TR-TR": runner._empty_evaluation_summary("not_run", "synthetic only"),
                    "TR-TS": _completed_summary(),
                    "TS-TR": _completed_summary(),
                    "TR+TS-TR": runner._empty_evaluation_summary("not_run", "synthetic only"),
                },
                [],
            )

        self.assertEqual(payload["run_mode"], "demo")

    def test_effective_parameters_record_origin(self):
        args = _parsed_runner_args([])

        self.assertEqual(args._effective_parameters["run_mode"]["origin"], "default")
        self.assertEqual(args._effective_parameters["execution_mode"]["origin"], "profile")
        self.assertIn("default_value", args._effective_parameters["execution_mode"])

    def test_resolved_config_records_effective_values_once(self):
        args = _parsed_runner_args(["--run_mode", "demo", "--pipeline", "synthetic"])
        resolved, _, _, _ = runner.resolve_config_for_command(
            args,
            {"model_type": "copy", "number_k_folds": 5},
            split_mode="provided",
            raw_root=Path("/tmp/raw"),
        )

        self.assertEqual(resolved.dataset.effective_number_k_folds, 1)
        self.assertEqual(resolved.sample_plan.generated_samples_per_class, 100)
        self.assertEqual(resolved.transform.feature_transform, args.feature_transform)
        self.assertIn("generated_samples_per_class", resolved.effective_parameters)

    def test_no_arguments_keep_full_campaign_behavior(self):
        args = _parsed_runner_args([])

        self.assertEqual(runner.choose_campaigns(args.campaign, full=True), runner.DEFAULT_CAMPAIGN)

    def test_campaign_sf_keeps_demo_campaign_behavior(self):
        args = _parsed_runner_args(["-c", "sf"])

        self.assertEqual(runner.choose_campaigns(args.campaign, full=False), runner.DEMO_CAMPAIGNS)

    def test_real_resample_sf_uses_single_control_campaign(self):
        args = _parsed_runner_args(["-c", "sf", "--synthetic_control", "real_resample"])
        campaigns = runner.choose_campaigns(args.campaign, full=False)

        self.assertEqual(runner.canonicalize_control_campaigns(campaigns, args), [runner.CONTROL_CAMPAIGN])

    def test_real_resample_preserves_explicit_legacy_campaign(self):
        args = _parsed_runner_args(["-c", "variational_demo", "--synthetic_control", "real_resample"])
        campaigns = runner.choose_campaigns(args.campaign, full=False)

        self.assertEqual(runner.canonicalize_control_campaigns(campaigns, args), ["variational_demo"])

    def test_control_campaign_runs_copy_model_and_records_control_artifact(self):
        campaign = runner.campaigns_available[runner.CONTROL_CAMPAIGN]
        params, values = zip(*campaign.items())
        combination = dict(zip(params, (value[0] for value in values)))

        self.assertEqual(combination["model_type"], "copy")
        self.assertEqual(combination["artifact_model_type"], "control")

    def test_legacy_csv_command_still_works(self):
        args = _parsed_runner_args(["--run_mode", "full", "--pipeline", "synthetic"])
        command = build_main_command(
            "python3",
            Path("/tmp/dataset.csv"),
            Path("/tmp/out"),
            {"model_type": "copy", "number_k_folds": 1, "classifier": "DecisionTree", "save_data": "True"},
            20,
            "continuous",
            -1,
            None,
            100,
            args,
        )

        self.assertIn("--data_load_path_file_input", command)
        self.assertNotIn("--data_format", command)
        self.assertEqual(command.count("--number_k_folds"), 1)
        self.assertEqual(command[command.index("--number_k_folds") + 1], "1")
        self.assertNotIn("--effective_number_k_folds", command)

    def test_provided_split_resolves_effective_folds_to_one(self):
        metadata = runner.resolve_k_fold_metadata({"number_k_folds": 5}, "provided")

        self.assertEqual(metadata["requested_number_k_folds"], 5)
        self.assertEqual(metadata["effective_number_k_folds"], 1)
        self.assertEqual(metadata["split_mode"], "provided")
        self.assertEqual(metadata["number_k_folds_origin"], "campaign")
        self.assertEqual(metadata["origin"], "campaign")

    def test_cross_validation_preserves_requested_folds(self):
        metadata = runner.resolve_k_fold_metadata({"number_k_folds": 5}, "cross_validation")

        self.assertEqual(metadata["requested_number_k_folds"], 5)
        self.assertEqual(metadata["effective_number_k_folds"], 5)
        self.assertEqual(metadata["split_mode"], "cross_validation")

    def test_cross_validation_command_preserves_requested_folds(self):
        args = _parsed_runner_args(["--run_mode", "full", "--pipeline", "synthetic"])
        command = build_main_command(
            "python3",
            Path("/tmp/dataset.csv"),
            Path("/tmp/out"),
            {"model_type": "copy", "number_k_folds": 5, "classifier": "DecisionTree", "save_data": "True"},
            20,
            "continuous",
            -1,
            None,
            100,
            args,
        )

        self.assertEqual(command.count("--number_k_folds"), 1)
        self.assertEqual(command[command.index("--number_k_folds") + 1], "5")

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

    def test_all_executes_tr_ts_ts_tr_and_tr_ts_tr(self):
        owner = _FakeEvaluationOwner("all")
        synthetic = SimpleNamespace(train_reader={"split": "train"}, test_reader={"split": "test"})

        run_synthetic_evaluation_modes(owner, {"x_evaluation_real": numpy.zeros((1, 2))}, synthetic)

        self.assertEqual(
            owner.calls,
            [
                ("guard", synthetic.test_reader),
                ("guard", synthetic.train_reader),
                ("TR-TS", synthetic.test_reader),
                ("TS-TR", synthetic.train_reader),
                ("TR+TS-TR", synthetic.train_reader),
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

    def test_provided_tr_ts_tr_uses_train_synthetic_train_and_test(self):
        bundle = _provided_bundle()
        owner = _FakeEvaluationOwner("tr_ts_tr", split_mode="provided")
        owner._dataset_bundle = bundle
        synthetic = SimpleNamespace(train_reader={"split": "synthetic_train"}, test_reader={"split": "synthetic_test"})

        run_synthetic_evaluation_modes(owner, {"dataset_bundle": bundle}, synthetic)

        self.assertEqual(owner.calls[-1], ("TR+TS-TR", "train", synthetic.train_reader, "test"))

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
        self.assertIn("--real_class_count_policy", command)
        self.assertEqual(command[command.index("--real_class_count_policy") + 1], "strict")
        self.assertIn("--samples_per_class_scope", command)
        self.assertEqual(command[command.index("--samples_per_class_scope") + 1], "split")
        self.assertEqual(command[command.index("--number_samples_per_class") + 1], build_number_samples_per_class_plan(28))

    def test_wrapper_help_lists_real_class_count_arguments(self):
        help_text = runner.build_parser().format_help()

        self.assertIn("--real_class_count_policy", help_text)
        self.assertIn("--samples_per_class_scope", help_text)

    def test_real_class_count_policy_strict_is_propagated(self):
        args = _batch_args()
        args.real_class_count_policy = "strict"

        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        self.assertEqual(command[command.index("--real_class_count_policy") + 1], "strict")

    def test_real_class_count_policy_uniform_min_is_propagated(self):
        args = _batch_args()
        args.real_class_count_policy = "uniform_min"

        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        self.assertEqual(command[command.index("--real_class_count_policy") + 1], "uniform_min")

    def test_samples_per_class_scope_split_is_propagated(self):
        args = _batch_args()
        args.samples_per_class_scope = "split"

        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        self.assertEqual(command[command.index("--samples_per_class_scope") + 1], "split")

    def test_real_class_count_cli_precedes_campaign(self):
        args = _batch_args()
        args.real_class_count_policy = "available_cap"

        command = _batch_command_for_args(
            self,
            args,
            {"model_type": "copy", "real_class_count_policy": "uniform_min"},
        )

        self.assertEqual(command[command.index("--real_class_count_policy") + 1], "available_cap")

    def test_real_class_count_arguments_are_not_duplicated(self):
        args = _batch_args()
        command = _batch_command_for_args(
            self,
            args,
            {
                "model_type": "copy",
                "real_class_count_policy": "uniform_min",
                "samples_per_class_scope": "split",
            },
        )

        self.assertEqual(command.count("--real_class_count_policy"), 1)
        self.assertEqual(command.count("--samples_per_class_scope"), 1)

    def test_strict_test_minimum_1539_accepts_request_500(self):
        labels = numpy.repeat(numpy.arange(APPCLASSNET_NUM_CLASSES, dtype=numpy.int64), 1539)

        indices, counts, short = runner.select_stratified_indices(
            numpy,
            labels,
            500,
            APPCLASSNET_NUM_CLASSES,
            seed=7,
            split_name="test",
            real_class_count_policy="strict",
            samples_per_class_scope="split",
        )

        self.assertEqual(indices.shape[0], APPCLASSNET_NUM_CLASSES * 500)
        self.assertFalse(short)
        self.assertEqual(min(counts.values()), 500)

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

    def test_vae_epochs_alias_is_forwarded_as_canonical_variational_epochs(self):
        args = _batch_args()
        args.vae_epochs = 30
        args._explicit_cli_options = {"vae_epochs"}

        command = _batch_command_for_args(
            self,
            args,
            {"model_type": "variational", "variational_autoencoder_number_epochs": 300},
        )

        self.assertNotIn("--vae_epochs", command)
        self.assertEqual(command.count("--variational_autoencoder_number_epochs"), 1)
        self.assertEqual(command[command.index("--variational_autoencoder_number_epochs") + 1], "30")

    def test_num_classes_subset_records_effective_num_classes(self):
        args = _batch_args()
        args.num_classes_subset = 10
        args._explicit_cli_options = {"num_classes_subset"}

        command = _batch_command_for_args(self, args, {"model_type": "copy"})
        manifest = runner.COMMAND_MANIFESTS[tuple(map(str, command))]

        self.assertNotIn("--num_classes_subset", command)
        self.assertEqual(command[command.index("--num_classes") + 1], "10")
        self.assertIn("batches/subsets", command[command.index("--train_x_path") + 1])
        self.assertEqual(manifest["resolved_config"]["dataset"]["effective_num_classes"], 10)
        self.assertEqual(
            manifest["resolved_config"]["effective_parameters"]["effective_num_classes"]["effective_value"],
            10,
        )

    def test_transform_arguments_are_preserved_separately(self):
        args = _batch_args()
        args.feature_transform = "minmax"
        args.generator_transform = "standard"
        args.classifier_transform = "preserve"
        args.evaluation_space = "transformed"

        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        self.assertEqual(command[command.index("--feature_transform") + 1], "minmax")
        self.assertEqual(command[command.index("--generator_transform") + 1], "standard")
        self.assertEqual(command[command.index("--classifier_transform") + 1], "preserve")
        self.assertEqual(command[command.index("--evaluation_space") + 1], "transformed")

    def test_unknown_main_flag_is_rejected_before_subprocess(self):
        command = _batch_command_for_args(self, _batch_args(), {"model_type": "copy"})
        command.extend(["--does_not_exist", "1"])

        with self.assertRaisesRegex(ValueError, "UnknownMainCommandArgument"):
            runner.validate_command_before_subprocess(command)

    def test_conflicting_classifier_aliases_raise_error(self):
        parser = runner.build_parser()
        args = parser.parse_args([
            "--batch_classifier",
            "sgd",
            "--eval_classifier",
            "decision_tree_subset",
        ])
        runner.annotate_explicit_cli_arguments(
            args,
            ["--batch_classifier", "sgd", "--eval_classifier", "decision_tree_subset"],
        )

        with self.assertWarns(DeprecationWarning), self.assertRaisesRegex(ValueError, "ConflictingClassifierArguments"):
            runner.normalize_classifier_arguments(args)

    def test_command_manifest_is_written_for_real_execution_only(self):
        args = _batch_args()
        command = _batch_command_for_args(self, args, {"model_type": "copy"})

        manifest_path = runner.write_command_manifest(command)

        self.assertTrue(manifest_path.is_file())
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["resolved_config"]["classifier"]["effective_classifier"], "decision_tree_subset")
        self.assertEqual(payload["transforms"]["feature_transform"], "preserve")
        self.assertEqual(payload["transforms"]["classifier_transform"], "preserve")
        self.assertEqual(payload["transforms"]["generator_transform"], "preserve")
        self.assertEqual(payload["transforms"]["evaluation_space"], "source")
        self.assertFalse(payload["transforms"]["transform_fitted"])
        self.assertFalse(payload["transforms"]["transform_applied"])
        self.assertIn("canonical_command", payload)

    def test_appclassnet_preserve_preprocessing_returns_source_root_and_train_only_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            raw_root.mkdir()
            for split_name, offset in (("train", 0.0), ("valid", 10.0), ("test", 20.0)):
                x_values = numpy.full((2, runner.APPCLASSNET_NUM_FEATURES), offset, dtype=numpy.float64)
                x_values[0, 0] = -0.5 + offset
                x_values[1, 1] = 0.5 + offset
                y_values = numpy.array([0, 1], dtype=numpy.int64)
                numpy.save(raw_root / f"{split_name}_x.npy", x_values)
                numpy.save(raw_root / f"{split_name}_y.npy", y_values)

            args = SimpleNamespace(
                source_profile="appclassnet_top200",
                feature_transform="preserve",
                generator_transform="preserve",
                classifier_transform="preserve",
                evaluation_space="source",
                allow_scaler_refit=False,
                allow_double_transform=False,
                inverse_transform_synthetic=True,
                use_mmap=True,
                execution_mode="batches",
                prepare_chunk_size=1,
                scaler="none",
                _preprocessing_policy=None,
            )

            with mock.patch.object(runner, "RESULTS_ROOT", root / "results"):
                effective_root, scaler_path, manifest_path = runner.preprocess_appclassnet_splits(
                    args,
                    raw_root,
                    "batches",
                )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            report = json.loads((Path(manifest_path).parent / "preprocessing_stats.json").read_text(encoding="utf-8"))
            scaler_exists = Path(scaler_path).is_file()
            scaled_train_exists = (Path(manifest_path).parent / "scaled_npy" / "train_x.npy").exists()

            self.assertEqual(effective_root, raw_root)
        self.assertTrue(scaler_exists)
        self.assertFalse(scaled_train_exists)
        self.assertEqual(manifest["feature_transform"], "preserve")
        self.assertEqual(manifest["evaluation_space"], "source")
        self.assertFalse(manifest["transform_fitted"])
        self.assertEqual(manifest["transform_parameters"], {})
        self.assertEqual(manifest["source_min"], -0.5)
        self.assertEqual(manifest["source_max"], 0.5)
        self.assertEqual(manifest["transformed_min"], -0.5)
        self.assertEqual(manifest["transformed_max"], 0.5)
        self.assertFalse(report["transform_applied"])

    def test_dryrun_run_cmd_does_not_call_subprocess_or_write_manifest(self):
        args = _batch_args()
        command = _batch_command_for_args(self, args, {"model_type": "copy"})
        manifest_path = Path(command[command.index("--output_dir") + 1]) / "command_manifest.json"
        previous_arguments = runner.arguments
        runner.arguments = SimpleNamespace(dryrun=True)
        self.addCleanup(setattr, runner, "arguments", previous_arguments)

        with mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("subprocess should not run")):
            runner.run_cmd(command)

        self.assertFalse(manifest_path.exists())

    def test_batch_classifier_metadata_records_requested_effective_and_fit_rows(self):
        x_values = numpy.array([[0.0], [1.0], [2.0], [3.0]], dtype=numpy.float32)
        y_values = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
        args = SimpleNamespace(
            random_state=0,
            train_samples_per_class=1,
            batch_classifier_subset_size=10,
            max_depth=2,
            decision_tree_criterion="gini",
            decision_tree_max_features=None,
            decision_tree_max_leaf_nodes=None,
        )

        _, metadata = train_batch_classifier(
            "decision_tree_subset",
            [(x_values, y_values)],
            num_classes=2,
            arguments=args,
        )

        self.assertEqual(metadata["requested_classifier"], "decision_tree_subset")
        self.assertEqual(metadata["effective_classifier"], "decision_tree_subset")
        self.assertEqual(metadata["classifier_name"], "DecisionTreeSubset")
        self.assertEqual(metadata["effective_fit_rows"], 2)

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
        self.assertNotIn("--batch_classifier", command)

    def test_batch_command_sends_only_effective_number_k_folds(self):
        command = _batch_command_for_args(
            self,
            _batch_args(),
            {"model_type": "copy", "number_k_folds": 2},
        )

        self.assertEqual(command.count("--number_k_folds"), 1)
        self.assertEqual(command[command.index("--number_k_folds") + 1], "1")
        self.assertNotIn("--effective_number_k_folds", command)

    def test_duplicate_command_arguments_are_rejected_before_subprocess(self):
        command = _batch_command_for_args(
            self,
            _batch_args(),
            {"model_type": "copy", "number_k_folds": 2},
        )
        command.extend(["--number_k_folds", "1"])

        with self.assertRaisesRegex(ValueError, "DuplicateCommandArgument"):
            runner.validate_command_before_subprocess(command)

    def test_main_accepts_produced_real_resample_command(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory) / "raw"
            output_dir = Path(directory) / "out"
            raw_root.mkdir()
            _write_npy_split(raw_root, "train", rows_per_class=2)
            _write_npy_split(raw_root, "valid", rows_per_class=1)
            _write_npy_split(raw_root, "test", rows_per_class=2)

            args = _batch_args(evaluation_mode="both")
            args.synthetic_control = "real_resample"
            args.train_samples_per_class = 2
            args.test_samples_per_class = 2
            args.synthetic_train_samples_per_class = 1
            args.synthetic_test_samples_per_class = 1
            args.generated_samples_per_class = 2
            args.save_synthetic_format = "npy_batches"

            command = build_batch_main_command(
                sys.executable,
                raw_root,
                output_dir,
                {"model_type": "copy", "number_k_folds": 2},
                20,
                "multiclass",
                args,
            )

            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
            combined_output = f"{completed.stdout}\n{completed.stderr}"

            self.assertNotEqual(completed.returncode, 2, combined_output)
            self.assertEqual(completed.returncode, 0, combined_output)
            self.assertNotIn("unrecognized arguments", combined_output)
            self.assertIn("Synthetic control source audit before selection", combined_output)
            self.assertIn("synthetic_batches/train/manifest.json", combined_output)
            self.assertIn("synthetic_batches/test/manifest.json", combined_output)

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
        self.assertIsNone(parsed.real_class_count_policy)
        self.assertIsNone(parsed.samples_per_class_scope)

    def test_evaluation_mode_none(self):
        _, payload = runner.write_batches_metrics(
            [],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="none"),
        )

        self.assertEqual(payload["TR-TR"]["status"], "not_run")
        self.assertEqual(payload["TR-TS"]["status"], "not_run")
        self.assertEqual(payload["TS-TR"]["status"], "not_run")
        self.assertEqual(payload["TR+TS-TR"]["status"], "not_run")
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
        self.assertEqual(payload["TS-TR"]["status"], "not_applicable")
        self.assertEqual(payload["TR+TS-TR"]["status"], "not_applicable")
        self.assertIsNotNone(payload["TR-TS"]["Accuracy"])

    def test_evaluation_mode_ts_tr(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=False, include_ts_tr=True)

        _, payload = runner.write_batches_metrics(
            [results_path],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="ts_tr"),
        )

        self.assertEqual(payload["TR-TS"]["status"], "not_applicable")
        self.assertEqual(payload["TS-TR"]["status"], "completed")
        self.assertEqual(payload["TR+TS-TR"]["status"], "not_applicable")
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
        self.assertEqual(payload["TR+TS-TR"]["status"], "not_applicable")
        self.assertIsNotNone(payload["TR-TS"]["MacroF1"])
        self.assertIsNotNone(payload["TS-TR"]["WeightedF1"])

    def test_evaluation_mode_tr_ts_tr(self):
        results_path = _write_batch_results_fixture(
            self,
            include_tr_ts=False,
            include_ts_tr=False,
            include_tr_ts_tr=True,
        )

        _, payload = runner.write_batches_metrics(
            [results_path],
            _temp_metrics_path(self),
            SimpleNamespace(evaluation_mode="tr_ts_tr"),
        )

        self.assertEqual(payload["TR-TS"]["status"], "not_applicable")
        self.assertEqual(payload["TS-TR"]["status"], "not_applicable")
        self.assertEqual(payload["TR+TS-TR"]["status"], "completed")
        self.assertIsNotNone(payload["TR+TS-TR"]["Accuracy"])

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
        self.assertEqual(payload["TR+TS-TR"]["status"], "not_run")

    def test_requested_evaluation_cannot_finish_not_run(self):
        results_path = _write_batch_results_fixture(self, include_tr_ts=False, include_ts_tr=False)

        with self.assertRaisesRegex(RuntimeError, "Requested evaluation TR-TS was not completed"):
            runner.write_batches_metrics(
                [results_path],
                _temp_metrics_path(self),
                SimpleNamespace(evaluation_mode="tr_ts"),
            )

    def test_requested_incomplete_evaluation_is_failed_not_not_run(self):
        summary = runner._build_method_summary(
            {"TR-TS": {}},
            "TR-TS",
            Path("/tmp/run"),
            {},
            active=True,
        )

        self.assertEqual(summary["status"], "failed")

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

    def evaluation_TR_TS_TR(
            self,
            dictionary_data=None,
            synthetic_data=None,
            *,
            real_train_data=None,
            synthetic_train_data=None,
            real_test_data=None):
        if real_train_data is not None or real_test_data is not None:
            if real_train_data.name != "train":
                raise EvaluationSplitMismatchError(
                    f"TR+TS-TR requires real split 'train', but received '{real_train_data.name}'."
                )
            if real_test_data.name != "test":
                raise EvaluationSplitMismatchError(
                    f"TR+TS-TR requires real split 'test', but received '{real_test_data.name}'."
                )
            self.calls.append(("TR+TS-TR", real_train_data.name, synthetic_train_data, real_test_data.name))
        else:
            self.calls.append(("TR+TS-TR", synthetic_data))

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


def _parsed_runner_args(raw_args):
    parser = runner.build_parser()
    args = parser.parse_args(raw_args)
    runner.annotate_explicit_cli_arguments(args, raw_args)
    runner.apply_execution_profile(args)
    runner.normalize_mmap_arguments(args)
    runner.normalize_classifier_arguments(args)
    return args


def _resolved_tr_tr_args(raw_args, validate=True):
    args = _parsed_runner_args(raw_args)
    campaigns = runner.choose_campaigns(args.campaign, full=args.run_mode_effective == "full")
    runner.normalize_preprocessing_arguments(args, campaigns)
    if args.baseline_real_only:
        sample_arguments = runner.resolve_effective_sample_arguments(args, {})
        for parameter, value in sample_arguments["values"].items():
            setattr(args, parameter, value)
        real_count_arguments = runner.resolve_effective_real_class_count_arguments(args, {})
        for parameter, value in real_count_arguments["values"].items():
            setattr(args, parameter, value)
    runner.normalize_explicit_tr_tr_sampling(args)
    if validate:
        runner.validate_tr_tr_cli_arguments(args)
    return args


def _completed_summary():
    summary = runner._empty_evaluation_summary("completed", None)
    summary.update({
        "classifier": "DecisionTreeSubset",
        "Accuracy": 1.0,
        "BalancedAccuracy": 1.0,
        "MacroF1": 1.0,
        "WeightedF1": 1.0,
    })
    return summary


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
        feature_transform="preserve",
        generator_transform="preserve",
        classifier_transform="preserve",
        evaluation_space="source",
        evaluation_mode=evaluation_mode,
        batch_classifier=None,
        use_mmap=True,
        max_train_samples=None,
        max_samples_per_class=None,
        train_samples_per_class=None,
        test_samples_per_class=None,
        synthetic_train_samples_per_class=None,
        synthetic_test_samples_per_class=None,
        generated_samples_per_class=None,
        real_class_count_policy=None,
        samples_per_class_scope=None,
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
        synthetic_control="none",
        class_subset=None,
        num_classes_subset=None,
        vae_epochs=None,
        gan_epochs=None,
        _effective_parameters={},
    )


def _batch_command_for_args(test_case, args, combination):
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    raw_root = Path(directory.name)
    for split in ("train", "valid", "test"):
        numpy.save(raw_root / f"{split}_x.npy", numpy.zeros((APPCLASSNET_NUM_CLASSES, 20), dtype=numpy.float32))
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


def _write_npy_split(raw_root, split, rows_per_class):
    labels = numpy.repeat(numpy.arange(APPCLASSNET_NUM_CLASSES, dtype=numpy.int64), rows_per_class)
    features = numpy.zeros((labels.shape[0], 20), dtype=numpy.float32)
    features[:, 0] = labels.astype(numpy.float32)
    features[:, 1] = numpy.arange(labels.shape[0], dtype=numpy.float32)
    numpy.save(raw_root / f"{split}_x.npy", features)
    numpy.save(raw_root / f"{split}_y.npy", labels)


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


def _write_batch_results_fixture(test_case, include_tr_ts=True, include_ts_tr=True, include_tr_ts_tr=False):
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
    if include_tr_ts_tr:
        results["TR+TS-TR"] = {"DecisionTreeSubset": {"1-Fold": _metric_block()}}
        results["BatchClassifier"]["1-Fold"]["TR+TS-TR"] = _metadata_block()
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
