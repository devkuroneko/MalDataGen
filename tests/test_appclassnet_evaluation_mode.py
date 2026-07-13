import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.Preprocessing.FeatureTransformManager import PreprocessingSpaceMismatchError
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
from main import run_synthetic_evaluation_modes
import run_appclassnet_top200 as runner
from run_appclassnet_top200 import APPCLASSNET_NUM_CLASSES
from run_appclassnet_top200 import build_batch_main_command
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

    def test_both_executes_both_with_same_synthetic_object(self):
        owner = _FakeEvaluationOwner("both")
        synthetic = {"same": object()}

        run_synthetic_evaluation_modes(owner, {"x_evaluation_real": numpy.zeros((1, 2))}, synthetic)

        self.assertEqual(owner.calls, [("guard", synthetic), ("TR-TS", synthetic), ("TS-TR", synthetic)])
        self.assertIs(owner.calls[1][1], owner.calls[2][1])

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
        self.assertEqual(command[command.index("--train_samples_per_class") + 1], "17")
        self.assertEqual(command[command.index("--test_samples_per_class") + 1], "11")
        self.assertIn("--synthetic_train_samples_per_class", command)
        self.assertIn("--synthetic_test_samples_per_class", command)


class _FakeEvaluationOwner:
    def __init__(self, evaluation_mode):
        self.arguments = SimpleNamespace(evaluation_mode=evaluation_mode)
        self.fold_number = 0
        self.calls = []

    def _guard_current_evaluation_space(self, dictionary_data, synthetic_data):
        self.calls.append(("guard", synthetic_data))

    def evaluation_TR_TS(self, dictionary_data, synthetic_data):
        self.calls.append(("TR-TS", synthetic_data))

    def evaluation_TS_TR(self, dictionary_data, synthetic_data):
        self.calls.append(("TS-TR", synthetic_data))

    def mark_evaluation_classifiers_not_applicable(self, evaluation_type, fold, reason):
        self.calls.append(("skip", evaluation_type))


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


if __name__ == "__main__":
    unittest.main()
