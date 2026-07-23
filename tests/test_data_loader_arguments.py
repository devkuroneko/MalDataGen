import unittest
from types import SimpleNamespace

from Engine.Arguments.ArgumentsDataLoader import add_argument_data_load
from Engine.Arguments.ArgumentsDataLoader import validate_data_load_arguments
from Engine.Arguments.Arguments import _configure_classifier_arguments
from Engine.Arguments.Arguments import _configure_pipeline_arguments
from Engine.Arguments.Arguments import _normalize_evaluation_protocol_arguments
from Engine.Arguments.ArgumentsFramework import add_argument_framework
from Engine.Arguments.Arguments import _normalize_preprocessing_arguments
from Engine.Evaluation.TrTrPipeline import build_tr_tr_classifier
from Engine.Evaluation.TrTrPipeline import tr_tr_config_from_namespace


def build_parser():
    parser = add_argument_framework()
    return add_argument_data_load(parser)


def parse_main_cli(raw_args):
    parsed = build_parser().parse_args(raw_args)
    parsed = _normalize_preprocessing_arguments(parsed)
    parsed = _configure_pipeline_arguments(parsed)
    parsed = _normalize_evaluation_protocol_arguments(parsed)
    parsed = _configure_classifier_arguments(parsed)
    return validate_data_load_arguments(parsed)


class DataLoaderArgumentsTest(unittest.TestCase):

    def test_legacy_command_defaults_to_csv_and_cross_validation(self):
        parser = build_parser()
        parsed = validate_data_load_arguments(parser.parse_args([]))

        self.assertEqual(parsed.data_format, "csv")
        self.assertEqual(parsed.split_mode, "cross_validation")
        self.assertEqual(parsed.real_class_count_policy, "strict")
        self.assertEqual(parsed.samples_per_class_scope, "split")
        self.assertIsNone(parsed.train_x_path)
        self.assertIsNone(parsed.train_y_path)
        self.assertEqual(parsed.data_load_path_file_input, "Datasets/converted/train_x.csv")
        self.assertEqual(parsed.number_samples_per_class, "1:256,2:256")
        self.assertFalse(parsed._legacy_number_samples_per_class_explicit)

    def test_main_help_lists_real_class_count_arguments(self):
        help_text = build_parser().format_help()

        self.assertIn("--real_class_count_policy", help_text)
        self.assertIn("--samples_per_class_scope", help_text)

    def test_npy_xy_with_train_paths_parses(self):
        parser = build_parser()
        parsed = validate_data_load_arguments(parser.parse_args([
            "--data_format", "npy_xy",
            "--train_x_path", "train_x.npy",
            "--train_y_path", "train_y.npy",
            "--target_type", "multiclass",
            "--feature_type", "continuous",
            "--num_classes", "200",
            "--sample_plan", "balanced_per_class",
            "--samples_per_class", "1000",
            "--mmap_npy",
        ]))

        self.assertEqual(parsed.data_format, "npy_xy")
        self.assertEqual(parsed.train_x_path, "train_x.npy")
        self.assertEqual(parsed.train_y_path, "train_y.npy")
        self.assertEqual(parsed.target_type, "multiclass")
        self.assertEqual(parsed.feature_type, "continuous")
        self.assertEqual(parsed.num_classes, 200)
        self.assertEqual(parsed.sample_plan, "balanced_per_class")
        self.assertEqual(parsed.samples_per_class, 1000)
        self.assertTrue(parsed.mmap_npy)

    def test_npy_xy_without_train_y_path_raises_clear_error(self):
        parser = build_parser()
        parsed = parser.parse_args([
            "--data_format", "npy_xy",
            "--train_x_path", "train_x.npy",
        ])

        with self.assertRaisesRegex(ValueError, "train_y_path"):
            validate_data_load_arguments(parsed)

    def test_split_mode_default_remains_cross_validation(self):
        parser = build_parser()
        parsed = validate_data_load_arguments(parser.parse_args([]))

        self.assertEqual(parsed.split_mode, "cross_validation")

    def test_appclassnet_profile_uses_strict_evaluation_protocol_by_default(self):
        parser = build_parser()
        parsed = parser.parse_args(["--source_profile", "appclassnet_top200"])

        normalized = _normalize_preprocessing_arguments(parsed)

        self.assertEqual(normalized.evaluation_protocol, "appclassnet_strict")
        self.assertEqual(normalized.feature_transform, "preserve")

    def test_legacy_profile_keeps_legacy_evaluation_protocol(self):
        parser = build_parser()
        parsed = parser.parse_args([])

        normalized = _normalize_preprocessing_arguments(parsed)

        self.assertEqual(normalized.evaluation_protocol, "legacy")

    def test_multiclass_num_classes_updates_legacy_model_class_defaults_for_npy(self):
        arguments = SimpleNamespace(
            data_format="npy_xy",
            train_x_path="train_x.npy",
            train_y_path="train_y.npy",
            valid_x_path=None,
            valid_y_path=None,
            test_x_path=None,
            test_y_path=None,
            split_mode="cross_validation",
            target_type="multiclass",
            num_classes=200,
            sample_plan="balanced_per_class",
            samples_per_class=1000,
            total_synthetic_rows=None,
            number_samples_per_class="1:256,2:256",
            autoencoder_number_classes=2,
            variational_autoencoder_number_classes=2,
            quantized_vae_number_classes=2,
            wasserstein_number_classes=2,
            wasserstein_gp_number_classes=2,
        )

        validated = validate_data_load_arguments(arguments)

        self.assertEqual(validated.autoencoder_number_classes, 200)
        self.assertEqual(validated.variational_autoencoder_number_classes, 200)
        self.assertEqual(validated.quantized_vae_number_classes, 200)
        self.assertEqual(validated.wasserstein_number_classes, 200)
        self.assertEqual(validated.wasserstein_gp_number_classes, 200)

    def test_provided_split_mode_allows_train_only_with_warning(self):
        parser = build_parser()
        parsed = parser.parse_args([
            "--data_format", "npy_xy",
            "--split_mode", "provided",
            "--sample_plan", "balanced_per_class",
            "--samples_per_class", "10",
            "--train_x_path", "train_x.npy",
            "--train_y_path", "train_y.npy",
        ])

        with self.assertLogs(level="WARNING") as logs:
            validated = validate_data_load_arguments(parsed)

        self.assertIs(validated, parsed)
        self.assertIn("train split only", "\n".join(logs.output))

    def test_npy_xy_without_explicit_sample_plan_raises_clear_error(self):
        parser = build_parser()
        parsed = parser.parse_args([
            "--data_format", "npy_xy",
            "--train_x_path", "train_x.npy",
            "--train_y_path", "train_y.npy",
        ])

        with self.assertRaisesRegex(ValueError, "explicit sampling plan"):
            validate_data_load_arguments(parsed)

    def test_valid_split_requires_x_y_pair(self):
        parser = build_parser()
        parsed = parser.parse_args([
            "--data_format", "npy_xy",
            "--sample_plan", "balanced_per_class",
            "--samples_per_class", "10",
            "--train_x_path", "train_x.npy",
            "--train_y_path", "train_y.npy",
            "--valid_x_path", "valid_x.npy",
        ])

        with self.assertRaisesRegex(ValueError, "valid_x_path"):
            validate_data_load_arguments(parsed)

    def test_main_parser_accepts_camel_decision_tree_for_tr_tr(self):
        parsed = parse_main_cli([
            "--pipeline", "tr_tr",
            "-c", "DecisionTree",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
        ])

        self.assertEqual(parsed.pipeline_effective, "tr_tr")
        self.assertEqual(parsed.baseline_classifier, "decision_tree")
        self.assertEqual(parsed.classifier_canonical, "DecisionTree")

    def test_main_parser_accepts_camel_random_forest_for_tr_tr(self):
        parsed = parse_main_cli([
            "--pipeline", "tr_tr",
            "-c", "RandomForest",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
        ])

        self.assertEqual(parsed.baseline_classifier, "random_forest")
        self.assertEqual(parsed.classifier_canonical, "RandomForest")

    def test_main_parser_accepts_snake_case_classifier_aliases(self):
        decision_tree = parse_main_cli([
            "--pipeline", "tr_tr",
            "-c", "decision_tree",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
        ])
        random_forest = parse_main_cli([
            "--pipeline", "tr_tr",
            "-c", "random_forest",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
        ])

        self.assertEqual(decision_tree.classifier_canonical, "DecisionTree")
        self.assertEqual(decision_tree.baseline_classifier, "decision_tree")
        self.assertEqual(random_forest.classifier_canonical, "RandomForest")
        self.assertEqual(random_forest.baseline_classifier, "random_forest")

    def test_use_mmap_and_mmap_npy_resolve_to_same_configuration(self):
        base = [
            "--pipeline", "tr_tr",
            "-c", "DecisionTree",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
        ]

        use_mmap = parse_main_cli([*base, "--use_mmap"])
        mmap_npy = parse_main_cli([*base, "--mmap_npy"])

        self.assertTrue(use_mmap.mmap_npy)
        self.assertTrue(use_mmap.use_mmap)
        self.assertTrue(mmap_npy.mmap_npy)
        self.assertTrue(mmap_npy.use_mmap)

    def test_pipeline_tr_tr_does_not_require_synthetic_arguments_or_sample_plan(self):
        parsed = parse_main_cli([
            "--pipeline", "tr_tr",
            "-c", "DecisionTree",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
        ])

        self.assertTrue(parsed.baseline_real_only)
        self.assertEqual(parsed.evaluation_protocol, "tr_tr")
        self.assertEqual(parsed.data_format, "npy_xy")
        self.assertEqual(parsed.split_mode, "provided")
        self.assertTrue(parsed.train_x_path.endswith("train_x.npy"))
        self.assertTrue(parsed.test_y_path.endswith("test_y.npy"))

    def test_balanced_per_class_requires_matching_quota(self):
        with self.assertRaisesRegex(ValueError, "train_samples_per_class"):
            parse_main_cli([
                "--pipeline", "tr_tr",
                "-c", "DecisionTree",
                "--source_profile", "appclassnet_top200",
                "--train_sampling", "balanced_per_class",
                "--test_sampling", "all",
            ])

    def test_rf_parameters_reach_random_forest_classifier(self):
        parsed = parse_main_cli([
            "--pipeline", "tr_tr",
            "-c", "RandomForest",
            "--source_profile", "appclassnet_top200",
            "--train_sampling", "all",
            "--test_sampling", "all",
            "--rf_n_estimators", "17",
            "--rf_n_jobs", "3",
        ])

        model = build_tr_tr_classifier(tr_tr_config_from_namespace(parsed))

        self.assertEqual(model.get_params(deep=True)["n_estimators"], 17)
        self.assertEqual(model.get_params(deep=True)["n_jobs"], 3)

    def test_csv_legacy_still_parses_without_pipeline(self):
        parsed = parse_main_cli([])

        self.assertEqual(parsed.data_format, "csv")
        self.assertEqual(parsed.pipeline_effective, "synthetic")
        self.assertFalse(parsed.baseline_real_only)


if __name__ == "__main__":
    unittest.main()
